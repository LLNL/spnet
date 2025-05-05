# Global imports
import numpy as np
import torch
import torchvision
from torchvision import transforms
import albumentations as albu
import cv2

# Package imports
from osr.engine import albu_transform as albu_fork


# xywh -> x1y1x2y2
def _bbox_transform(bboxes):
    if len(bboxes) == 0:
        bboxes = torch.empty((0, 4), dtype=torch.float)
    else:
        bboxes = torch.FloatTensor(bboxes)
        bboxes[:, 2:] += bboxes[:, :2]
    return bboxes


# Standardize data
def _normalize(image, mean, std):
    image = image / 255.0
    mean = np.array(mean)
    std = np.array(std)
    return ((image - mean[None, None, :]) / std[None, None, :]).astype(np.float32)


# 
def to_uint8(image):
    #print('to_uint8 before:', image.dtype, image.min(), image.max())
    # Subtract min
    minval = image.min()
    image = image - minval
    # Divide by max
    maxval = image.max()
    image = image / maxval
    # Multiply by 255.0
    image = image * 255.0
    # Convert to uint8
    image = image.astype(np.uint8)
    #print('to_uint8 after:', image.dtype, image.min(), image.max())
    # Return uint8 image
    return image, minval, maxval

#
def from_uint8(image, minval, maxval):
    #print('from_uint8 before:', image.dtype, image.min(), image.max())
    # Convert to float32
    image = image.astype(np.float32)
    # Divide by 255.0
    image = image / 255.0
    # Multiply by max
    image = image * maxval
    # Add min
    image = image + minval
    #print('from_uint8 after:', image.dtype, image.min(), image.max())
    # Return float32 image
    return image
    

# Wrapper class for handling interface with albumentations
class AlbuWrapper(object):
    def __init__(self, albu_transform, stat_dict):
        self.albu_transform = albu_transform
        self.stat_dict = stat_dict
        self.img_transform = torchvision.transforms.ToTensor()
        self.to_key_dict = {
            'boxes': 'bboxes',
            'labels': 'category_ids',
        }
        self.from_key_dict = {v:k for k,v in self.to_key_dict.items()}
        self.from_transform_dict = {
            'category_ids': torch.LongTensor,
            'person_id': torch.LongTensor,
            'bboxes': _bbox_transform,
            'iou_thresh': torch.FloatTensor,
            'id': torch.LongTensor,
            'is_known': torch.BoolTensor,
        }

    def __call__(self, data): 
        # Wrap data into format for albumentations
        image, target = data
        
        # Make sure incoming dimensions match
        assert target['boxes'].size(0) == target['labels'].size(0) == len(target['person_id']), 'Incoming augmentation dimension mismatch'

        #
        rekeyed_target = {(self.to_key_dict[k] if k in self.to_key_dict else k):v for k,v in target.items()}

        #
        #print('before:', _normalize(np.array(image), self.stat_dict['mean'], self.stat_dict['std']).dtype, _normalize(np.array(image), self.stat_dict['mean'], self.stat_dict['std']).min(), _normalize(np.array(image), self.stat_dict['mean'], self.stat_dict['std']).max())
        uint8_image, minval, maxval = to_uint8(_normalize(np.array(image), self.stat_dict['mean'], self.stat_dict['std']))
        albu_result = self.albu_transform(image=uint8_image, **rekeyed_target)
        albu_result['image'] = from_uint8(albu_result['image'], minval, maxval)
        #print('after:', albu_result['image'].dtype, albu_result['image'].min(), albu_result['image'].max())
        #exit()
        new_image = self.img_transform(albu_result['image'])
        new_target = {}
        for k,v in albu_result.items():
            if k != 'image':
                new_target[self.from_key_dict[k] if k in self.from_key_dict else k] = self.from_transform_dict[k](v) if k in self.from_transform_dict else v

        #
        return new_image, new_target

# Wrapper class for handling interface with albumentations
class AlbuWrapper2(object):
    def __init__(self, albu_transform, stat_dict):
        self.albu_transform = albu_transform
        self.stat_dict = stat_dict
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=stat_dict['mean'], std=stat_dict['std']),
        ])
        self.to_key_dict = {
            'boxes': 'bboxes',
            'labels': 'category_ids',
        }
        self.from_key_dict = {v:k for k,v in self.to_key_dict.items()}
        self.from_transform_dict = {
            'category_ids': torch.LongTensor,
            'person_id': torch.LongTensor,
            'bboxes': _bbox_transform,
            'iou_thresh': torch.FloatTensor,
            'id': torch.LongTensor,
            'is_known': torch.BoolTensor,
        }

    def __call__(self, data): 
        # Wrap data into format for albumentations
        image, target = data
        
        # Make sure incoming dimensions match
        assert target['boxes'].size(0) == target['labels'].size(0) == len(target['person_id']), 'Incoming augmentation dimension mismatch'

        #
        rekeyed_target = {(self.to_key_dict[k] if k in self.to_key_dict else k):v for k,v in target.items()}

        #
        albu_result = self.albu_transform(image=np.array(image), **rekeyed_target)
        new_image = self.img_transform(albu_result['image'])
        new_target = {}
        for k,v in albu_result.items():
            if k != 'image':
                new_target[self.from_key_dict[k] if k in self.from_key_dict else k] = self.from_transform_dict[k](v) if k in self.from_transform_dict else v

        #
        return new_image, new_target

# Random Resized Crop augmentation
def get_transform_lursc(train, stat_dict, min_size=900, max_size=1500, crop_res=512,
        rfc_prob=1.0, rsc_prob=1.0, rbsc_er=0.0):
    if train:
        transform_list = [
            albu.BBoxSafeRandomCrop(p=1.0),
            albu.LongestMaxSize(max_size=512, p=1.0),
            albu.PadIfNeeded(min_width=512, min_height=512, border_mode=cv2.BORDER_CONSTANT, value=0.0),
            albu_fork.LogUniformResize(min_size=256, max_size=512, p=1),
            albu.HorizontalFlip(p=0.5),
        ]
        albu_transform = albu.Compose(transform_list,
            bbox_params=albu.BboxParams(
                format='coco', label_fields=['category_ids', 'person_id', 'id', 'iou_thresh', 'is_known'],
                min_visibility=0.4, min_area=16,
            )
        )    
    else:
        transform_list = [albu_fork.WindowResize(min_size=min_size, max_size=max_size)] 
        albu_transform = albu.Compose(transform_list,
            bbox_params=albu.BboxParams(
                format='coco', label_fields=['category_ids', 'person_id', 'id', 'iou_thresh', 'is_known'],
                min_visibility=0.4,
            )
        )    
    albu_transform_dict = {'test': albu_transform}
    return AlbuWrapper(albu_transform, stat_dict)

# Random Resized Crop augmentation
def get_transform_rrc_scale(train, stat_dict, crop_res=512,
        rfc_prob=1.0, rsc_prob=1.0, rbsc_er=0.0):
    # Calculate padding value
    mean_arr = np.array(stat_dict['mean'])
    std_arr = np.array(stat_dict['std'])
    pad_arr = - mean_arr / std_arr
    transform_list = [] 
    if train:
        transform_list = [
            albu.RandomScale(scale_limit=(0.5, 2.0), p=1),
            albu.PadIfNeeded(min_width=4*crop_res, min_height=4*crop_res, border_mode=cv2.BORDER_CONSTANT, value=pad_arr),
            albu.OneOf([
                albu_fork.RandomFocusedCrop2(height=crop_res, width=crop_res, p=rfc_prob),
                albu.RandomSizedBBoxSafeCrop(crop_res, crop_res, erosion_rate=rbsc_er, interpolation=1, p=rsc_prob),
            ], p=1),
            albu.HorizontalFlip(p=0.5),
        ]
    albu_transform = albu.Compose(transform_list,
        bbox_params=albu.BboxParams(
            format='coco', label_fields=['category_ids', 'person_id', 'id', 'iou_thresh', 'is_known'],
            min_visibility=0.4,
        )
    )    
    albu_transform_dict = {'test': albu_transform}
    return AlbuWrapper(albu_transform, stat_dict)


# Random Resized Crop augmentation
def get_transform_rrc(train, stat_dict, min_size=900, max_size=1500, crop_res=512,
        rfc_prob=1.0, rsc_prob=1.0, rbsc_er=0.0):
    transform_list = [albu_fork.WindowResize(min_size=min_size, max_size=max_size)] 
    if train:
        transform_list = [
            #albu.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.5),
            albu_fork.WindowResize(min_size=min_size, max_size=max_size),
            albu.PadIfNeeded(min_width=crop_res, min_height=crop_res, border_mode=cv2.BORDER_CONSTANT, value=0.0),
            albu.OneOf([
                albu_fork.RandomFocusedCrop(height=crop_res, width=crop_res, p=rfc_prob),
                albu.RandomSizedBBoxSafeCrop(crop_res, crop_res, erosion_rate=rbsc_er, interpolation=1, p=rsc_prob),
            ], p=1),
            albu.HorizontalFlip(p=0.5),
            #albu_fork.FractionDropout(min_frac_holes=0.5, min_width=4, min_height=4, max_width=4, max_height=4, p=1)
            #albu_fork.RemoveBackground(p=0.5)
        ]
    albu_transform = albu.Compose(transform_list,
        bbox_params=albu.BboxParams(
            format='coco', label_fields=['category_ids', 'person_id', 'id', 'iou_thresh', 'is_known'],
            min_visibility=0.4,
        )
    )    
    albu_transform_dict = {'test': albu_transform}
    return AlbuWrapper(albu_transform, stat_dict)

# Random Resized Crop augmentation
def get_transform_rrcj(train, stat_dict, min_size=900, max_size=1500, crop_res=512,
        rfc_prob=1.0, rsc_prob=1.0, rbsc_er=0.0):
    transform_list = [albu_fork.WindowResize(min_size=min_size, max_size=max_size)] 
    if train:
        transform_list = [
            albu_fork.WindowResize(min_size=min_size, max_size=max_size),
            albu.PadIfNeeded(min_width=crop_res, min_height=crop_res, border_mode=cv2.BORDER_CONSTANT, value=0.0),
            albu.OneOf([
                albu_fork.RandomFocusedCrop(height=crop_res, width=crop_res, p=rfc_prob),
                albu.RandomSizedBBoxSafeCrop(crop_res, crop_res, erosion_rate=rbsc_er, interpolation=1, p=rsc_prob),
            ], p=1),
            ### Color / Blur
            #albu.ColorJitter(
            #    brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
            #albu.ToGray(p=0.2),
            albu.GaussianBlur(blur_limit=0, sigma_limit=(0.1, 2), p=0.5),
            # Flip
            albu.HorizontalFlip(p=0.5),
        ]
    albu_transform = albu.Compose(transform_list,
        bbox_params=albu.BboxParams(
            format='coco', label_fields=['category_ids', 'person_id', 'id', 'iou_thresh', 'is_known'],
            min_visibility=0.4,
        )
    )    
    albu_transform_dict = {'test': albu_transform}
    return AlbuWrapper(albu_transform, stat_dict)

# Random Resized Crop augmentation
def get_transform_rsc(train, stat_dict, min_size=900, max_size=1500, crop_res=512,
        rfc_prob=1.0, rsc_prob=1.0, rbsc_er=0.0):
    transform_list = [albu_fork.WindowResize(min_size=min_size, max_size=max_size)] 
    if train:
        transform_list = [
            albu_fork.WindowResize(min_size=min_size, max_size=max_size),
            albu.PadIfNeeded(min_width=crop_res, min_height=crop_res, border_mode=cv2.BORDER_CONSTANT, value=0.0),
                albu.RandomResizedCrop(crop_res, crop_res, scale=(0.25, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=1, p=rfc_prob),
                albu.RandomSizedBBoxSafeCrop(crop_res, crop_res, erosion_rate=rbsc_er, interpolation=1, p=rsc_prob),
            albu.HorizontalFlip(p=0.5),
        ]
    albu_transform = albu.Compose(transform_list,
        bbox_params=albu.BboxParams(
            format='coco', label_fields=['category_ids', 'person_id', 'id', 'iou_thresh', 'is_known'],
            min_visibility=0.4,
        )
    )    
    albu_transform_dict = {'test': albu_transform}
    return AlbuWrapper(albu_transform, stat_dict)


# Random Resized Crop augmentation
def get_transform_rrc2(train, stat_dict, min_size=900, max_size=1500, crop_res=512,
        rfc_prob=1.0, rsc_prob=1.0, rbsc_er=0.0):
    transform_list = [albu_fork.WindowResize(min_size=min_size, max_size=max_size)] 
    if train:
        # Calculate padding value
        mean_arr = np.array(stat_dict['mean'])
        std_arr = np.array(stat_dict['std'])
        pad_arr = - mean_arr / std_arr
        #
        transform_list = [
            albu_fork.WindowResize(min_size=min_size, max_size=max_size),
            albu.PadIfNeeded(min_width=3*crop_res, min_height=3*crop_res, border_mode=cv2.BORDER_CONSTANT, value=pad_arr),
            albu.OneOf([
                albu_fork.RandomFocusedCrop2(height=crop_res, width=crop_res, p=rfc_prob),
                albu.RandomSizedBBoxSafeCrop(crop_res, crop_res, erosion_rate=rbsc_er, interpolation=1, p=rsc_prob),
            ], p=1),
            albu.HorizontalFlip(p=0.5),
        ]
    albu_transform = albu.Compose(transform_list,
        bbox_params=albu.BboxParams(
            format='coco', label_fields=['category_ids', 'person_id', 'id', 'iou_thresh', 'is_known'],
            min_visibility=0.4,
        )
    )    
    albu_transform_dict = {'test': albu_transform}
    return AlbuWrapper(albu_transform, stat_dict)

# Random Resized Crop augmentation
def get_transform_lmspad(train, stat_dict, min_size=900, max_size=1500, crop_res=512,
        rfc_prob=1.0, rsc_prob=1.0, rbsc_er=0.0):
    # Calculate padding value
    mean_arr = np.array(stat_dict['mean'])
    std_arr = np.array(stat_dict['std'])
    pad_arr = - mean_arr / std_arr
    #
    transform_list = [
        albu.LongestMaxSize(max_size=crop_res, p=1.0),
        albu.PadIfNeeded(min_width=crop_res, min_height=crop_res, border_mode=cv2.BORDER_CONSTANT, value=pad_arr),
    ]

    # Compose
    albu_transform = albu.Compose(transform_list,
        bbox_params=albu.BboxParams(
            format='coco', label_fields=['category_ids', 'person_id', 'id', 'iou_thresh', 'is_known'],
            min_visibility=0.4,
        )
    )    
    albu_transform_dict = {'test': albu_transform}
    return AlbuWrapper(albu_transform, stat_dict)

# Window Resize augmentation
def get_transform_lurrc(train, stat_dict, min_size=900, max_size=1500, crop_res=512,
        rfc_prob=1.0, rsc_prob=1.0, rbsc_er=0.0):
    bbox_params = albu.BboxParams(
        format='coco', label_fields=['category_ids', 'person_id', 'id', 'iou_thresh', 'is_known'],
        min_visibility=0.4,
    )
    train_bbox_params = albu.BboxParams(
        format='coco', label_fields=['category_ids', 'person_id', 'id', 'iou_thresh', 'is_known'],
        min_visibility=0.4, min_area=16
    )
    albu_transform = albu.Compose([albu_fork.WindowResize(min_size=min_size, max_size=max_size)], bbox_params=bbox_params) 
    if train:
        albu_transform = albu.OneOf([
            albu.Compose([
                albu.BBoxSafeRandomCrop(p=1.0),
                albu.LongestMaxSize(max_size=512, p=1.0),
                albu.PadIfNeeded(min_width=512, min_height=512, border_mode=cv2.BORDER_CONSTANT, value=0.0),
                albu_fork.LogUniformResize(min_size=256, max_size=512, p=1),
                albu.HorizontalFlip(p=0.5),
            ], bbox_params=train_bbox_params),
            albu.Compose([
                albu_fork.WindowResize(min_size=min_size, max_size=max_size),
                albu.PadIfNeeded(min_width=crop_res, min_height=crop_res, border_mode=cv2.BORDER_CONSTANT, value=0.0),
                albu.OneOf([
                    albu_fork.RandomFocusedCrop(height=crop_res, width=crop_res, p=rfc_prob),
                    albu.RandomSizedBBoxSafeCrop(crop_res, crop_res, erosion_rate=rbsc_er, interpolation=1, p=rsc_prob),
                ], p=1),
                albu.HorizontalFlip(p=0.5),
            ], bbox_params=train_bbox_params)
        ], p=1)
    albu_transform_dict = {'test': albu_transform}
    return AlbuWrapper(albu_transform, stat_dict)

# Window Resize augmentation
def get_transform_wrsrrc(train, stat_dict, min_size=900, max_size=1500, crop_res=512,
        rfc_prob=1.0, rsc_prob=1.0, rbsc_er=0.0):
    bbox_params = albu.BboxParams(
        format='coco', label_fields=['category_ids', 'person_id', 'id', 'iou_thresh', 'is_known'],
        min_visibility=0.4,
    )
    train_bbox_params = albu.BboxParams(
        format='coco', label_fields=['category_ids', 'person_id', 'id', 'iou_thresh', 'is_known'],
        min_visibility=0.4, min_area=16
    )
    albu_transform = albu.Compose([albu_fork.WindowResize(min_size=min_size, max_size=max_size)], bbox_params=bbox_params)
    if train:
        albu_transform = albu.OneOf([
            albu.Compose([
                albu_fork.WindowResize(min_size=min_size, max_size=max_size),
                albu.HorizontalFlip(p=0.5),
            ], bbox_params=train_bbox_params),
            albu.Compose([
                albu_fork.WindowResize(min_size=min_size, max_size=max_size),
                albu.PadIfNeeded(min_width=crop_res, min_height=crop_res, border_mode=cv2.BORDER_CONSTANT, value=0.0),
                albu.OneOf([
                    albu_fork.RandomFocusedCrop(height=crop_res, width=crop_res, p=rfc_prob),
                    albu.RandomSizedBBoxSafeCrop(crop_res, crop_res, erosion_rate=rbsc_er, interpolation=1, p=rsc_prob),
                ], p=1),
                albu.HorizontalFlip(p=0.5),
            ], bbox_params=train_bbox_params)
        ], p=1)
    albu_transform_dict = {'test': albu_transform}
    return AlbuWrapper(albu_transform, stat_dict)

# Window Resize augmentation
def get_transform_rwrsrsc(train, stat_dict, min_size=900, max_size=1500, crop_res=512,
        rfc_prob=1.0, rsc_prob=1.0, rbsc_er=0.0):
    bbox_params = albu.BboxParams(
        format='coco', label_fields=['category_ids', 'person_id', 'id', 'iou_thresh', 'is_known'],
        min_visibility=0.4,
    )
    albu_transform = albu.Compose([albu_fork.WindowResize(min_size=min_size, max_size=max_size)], bbox_params=bbox_params) 
    if train:
        albu_transform = albu.OneOf([
            albu.Compose([
                albu_fork.WindowResize(min_size=min_size, max_size=max_size),
                albu.PadIfNeeded(min_width=crop_res, min_height=crop_res, border_mode=cv2.BORDER_CONSTANT, value=0.0),
                albu.RandomSizedBBoxSafeCrop(crop_res, crop_res, erosion_rate=rbsc_er, interpolation=1, p=rsc_prob),
                albu.HorizontalFlip(p=0.5),
            ], bbox_params=bbox_params),
            albu.Compose([
                albu.BBoxSafeRandomCrop(p=1.0),
                albu_fork.WindowResize(min_size=min_size, max_size=max_size),
                albu.HorizontalFlip(p=0.5),
            ], bbox_params=bbox_params)
        ], p=1)
    albu_transform_dict = {'test': albu_transform}
    return AlbuWrapper(albu_transform, stat_dict)

# Window Resize augmentation
def get_transform_rwrs(train, stat_dict, min_size=900, max_size=1500):
    transform_list = [albu_fork.WindowResize(min_size=min_size, max_size=max_size)] 
    if train:
        transform_list = [
            albu.BBoxSafeRandomCrop(p=1.0),
            albu_fork.WindowResize(min_size=min_size, max_size=max_size),
            albu.HorizontalFlip(p=0.5),
        ]
    albu_transform = albu.Compose(transform_list,
        bbox_params=albu.BboxParams(
            format='coco', label_fields=['category_ids', 'person_id', 'id', 'iou_thresh', 'is_known'],
            min_visibility=0.4,
        )
    )    
    albu_transform_dict = {'test': albu_transform}
    return AlbuWrapper(albu_transform, stat_dict)


# Window Resize augmentation
def get_transform_wrs(train, stat_dict, min_size=900, max_size=1500):
    transform_list = [albu_fork.WindowResize(min_size=min_size, max_size=max_size)] 
    if train:
        transform_list = [
            albu_fork.WindowResize(min_size=min_size, max_size=max_size),
            albu.HorizontalFlip(p=0.5),
        ]
    albu_transform = albu.Compose(transform_list,
        bbox_params=albu.BboxParams(
            format='coco', label_fields=['category_ids', 'person_id', 'id', 'iou_thresh', 'is_known'],
            min_visibility=0.4,
        )
    )    
    albu_transform_dict = {'test': albu_transform}
    return AlbuWrapper(albu_transform, stat_dict)

# Resize augmentation
def get_transform_rs(train, stat_dict, height=512, width=512):
    transform_list = [albu.Resize(height, width)]
    albu_transform = albu.Compose(transform_list,
        bbox_params=albu.BboxParams(
            format='coco', label_fields=['category_ids', 'person_id', 'id', 'iou_thresh', 'is_known'],
            min_visibility=0.4,
        )
    )    
    albu_transform_dict = {'test': albu_transform}
    return AlbuWrapper(albu_transform, stat_dict)

# Window Resize augmentation
def get_transform_in1k(train, stat_dict,
        train_resize_size=256, train_crop_size=176,
        val_resize_size=232, val_crop_size=224):
    if train:
        transform_list = [
            #albu.RandomResizedCrop(train_crop_size, train_crop_size,
            #    scale=(0.25, 1.0), ratio=(0.75, 1.33), 
            #    interpolation=1, p=1),
            albu.RandomSizedBBoxSafeCrop(train_crop_size, train_crop_size),
            albu.OneOf([
                # Identity
                albu.Affine(p=1),
                # ShearX (REDUCED for bboxes)
                albu.Affine(p=1, shear={'x': 45, 'y': 0}),
                # ShearY (REDUCED for bboxes)
                albu.Affine(p=1, shear={'x': 0, 'y': 45}),
                # TranslateX
                albu.Affine(p=1, translate_px={'x': 32, 'y': 0}),
                # TranslateY
                albu.Affine(p=1, translate_px={'x': 0, 'y': 32}),
                # Rotate (CHANGED to Rotate90)
                albu.RandomRotate90(p=1),
                # Brightness
                albu.RandomBrightness(limit=0.99, p=1),
                # Color
                albu.HueSaturationValue(hue_shift_limit=0,
                    sat_shift_limit=255,
                    val_shift_limit=0, p=1),
                # Contrast
                albu.RandomContrast(limit=0.99, p=1),
                # Sharpness (MODIFIED -- no analog)
                albu.Sharpen(alpha=(0.0, 0.99), lightness=(0.0, 0.99), p=1),
                # Posterize (MODIFIED -- confusing)
                albu.Posterize(num_bits=8, p=1),
                # Solarize
                albu.Solarize(threshold=255, p=1),
                # AutoContrast (CHANGED to CLAHE)
                albu.CLAHE(p=1),
                # Equalize
                albu.Equalize(p=1, mode='pil'),
            ], p=1),
            albu.HorizontalFlip(p=0.5),
        ]
    else:
        transform_list = [
            albu.SmallestMaxSize(val_resize_size),
            albu.CenterCrop(val_crop_size, val_crop_size),
        ]
    albu_transform = albu.Compose(transform_list,
        bbox_params=albu.BboxParams(
            format='coco', label_fields=['category_ids', 'person_id', 'id', 'iou_thresh', 'is_known'],
            min_visibility=1.0,
        )
    )    
    albu_transform_dict = {'test': albu_transform}
    return AlbuWrapper2(albu_transform, stat_dict)

# Window Resize augmentation
def get_transform_in1k2(train, stat_dict,
        train_resize_size=256, train_crop_size=176,
        val_resize_size=232, val_crop_size=224, min_size=900, max_size=1500):
    if train:
        transform_list = [
            albu.RandomSizedBBoxSafeCrop(train_crop_size, train_crop_size),
            albu.OneOf([
                # Identity
                albu.Affine(p=1),
                # ShearX (REDUCED for bboxes)
                albu.Affine(p=1, shear={'x': 45, 'y': 0}),
                # ShearY (REDUCED for bboxes)
                albu.Affine(p=1, shear={'x': 0, 'y': 45}),
                # TranslateX
                albu.Affine(p=1, translate_px={'x': 32, 'y': 0}),
                # TranslateY
                albu.Affine(p=1, translate_px={'x': 0, 'y': 32}),
                # Rotate (CHANGED to Rotate90)
                albu.RandomRotate90(p=1),
                # Brightness
                albu.RandomBrightness(limit=0.99, p=1),
                # Color
                albu.HueSaturationValue(hue_shift_limit=0,
                    sat_shift_limit=255,
                    val_shift_limit=0, p=1),
                # Contrast
                albu.RandomContrast(limit=0.99, p=1),
                # Sharpness (MODIFIED -- no analog)
                albu.Sharpen(alpha=(0.0, 0.99), lightness=(0.0, 0.99), p=1),
                # Posterize (MODIFIED -- confusing)
                albu.Posterize(num_bits=8, p=1),
                # Solarize
                albu.Solarize(threshold=255, p=1),
                # AutoContrast (CHANGED to CLAHE)
                albu.CLAHE(p=1),
                # Equalize
                albu.Equalize(p=1, mode='pil'),
            ], p=1),
            albu.HorizontalFlip(p=0.5),
        ]
    else:
        transform_list = [albu_fork.WindowResize(min_size=min_size, max_size=max_size)] 
    albu_transform = albu.Compose(transform_list,
        bbox_params=albu.BboxParams(
            format='coco', label_fields=['category_ids', 'person_id', 'id', 'iou_thresh', 'is_known'],
            min_visibility=0.9,
        )
    )    
    albu_transform_dict = {'test': albu_transform}
    return AlbuWrapper2(albu_transform, stat_dict)

# Window Resize augmentation
def get_transform_in1k3(train, stat_dict, min_size=900, max_size=1500, crop_res=512,
        rfc_prob=1.0, rsc_prob=1.0, rbsc_er=0.0):
    if train:
        transform_list = [
            albu_fork.WindowResize(min_size=min_size, max_size=max_size),
            albu.PadIfNeeded(min_width=crop_res, min_height=crop_res, border_mode=cv2.BORDER_CONSTANT, value=0.0),
            albu.OneOf([
                albu_fork.RandomFocusedCrop(height=crop_res, width=crop_res, p=rfc_prob),
                albu.RandomSizedBBoxSafeCrop(crop_res, crop_res, erosion_rate=rbsc_er, interpolation=1, p=rsc_prob),
            ], p=1),
            albu.OneOf([
                # Identity
                albu.Affine(p=1),
                # ShearX (REDUCED for bboxes)
                albu.Affine(p=1, shear={'x': 45, 'y': 0}),
                # ShearY (REDUCED for bboxes)
                albu.Affine(p=1, shear={'x': 0, 'y': 45}),
                # TranslateX
                albu.Affine(p=1, translate_px={'x': 32, 'y': 0}),
                # TranslateY
                albu.Affine(p=1, translate_px={'x': 0, 'y': 32}),
                # Rotate (CHANGED to Rotate90)
                albu.RandomRotate90(p=1),
                # Brightness
                albu.RandomBrightness(limit=0.99, p=1),
                # Color
                albu.HueSaturationValue(hue_shift_limit=0,
                    sat_shift_limit=255,
                    val_shift_limit=0, p=1),
                # Contrast
                albu.RandomContrast(limit=0.99, p=1),
                # Sharpness (MODIFIED -- no analog)
                albu.Sharpen(alpha=(0.0, 0.99), lightness=(0.0, 0.99), p=1),
                # Posterize (MODIFIED -- confusing)
                albu.Posterize(num_bits=8, p=1),
                # Solarize
                albu.Solarize(threshold=255, p=1),
                # AutoContrast (CHANGED to CLAHE)
                albu.CLAHE(p=1),
                # Equalize
                albu.Equalize(p=1, mode='pil'),
            ], p=1),
            albu.HorizontalFlip(p=0.5),
        ]
    else:
        transform_list = [albu_fork.WindowResize(min_size=min_size, max_size=max_size)] 
    albu_transform = albu.Compose(transform_list,
        bbox_params=albu.BboxParams(
            format='coco', label_fields=['category_ids', 'person_id', 'id', 'iou_thresh', 'is_known'],
            min_visibility=0.4,
        )
    )    
    albu_transform_dict = {'test': albu_transform}
    return AlbuWrapper2(albu_transform, stat_dict)

# Random Resized Crop augmentation
def get_transform_rrcin1k(train, stat_dict, min_size=900, max_size=1500, crop_res=512,
        rfc_prob=1.0, rsc_prob=1.0, rbsc_er=0.0):
    transform_list = [albu_fork.WindowResize(min_size=min_size, max_size=max_size)] 
    if train:
        transform_list = [
            ### in1k (without translation)
            albu.OneOf([
                # Identity
                albu.Affine(p=1, scale=1, rotate=0, shear=0),
                # ShearX (REDUCED for bboxes)
                albu.Affine(p=1, scale=1, rotate=0, shear={'x': 0, 'y': (-15, 15)}),
                # ShearY (REDUCED for bboxes)
                albu.Affine(p=1, scale=1, rotate=0, shear={'x': (-15, 15), 'y': 0}),
                # Rotate
                albu.Rotate(p=1, limit=(-15, 15)),
                # Rotate (CHANGED to Rotate90)
                albu.RandomRotate90(p=1),
                # Brightness
                albu.RandomBrightness(limit=0.2, p=1),
                # Color
                albu.HueSaturationValue(hue_shift_limit=0,
                    sat_shift_limit=30,
                    val_shift_limit=0, p=1),
                # Contrast
                albu.RandomContrast(limit=0.2, p=1),
                # Sharpness (MODIFIED -- no analog)
                albu.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1),
                # Posterize (MODIFIED -- confusing)
                albu.Posterize(num_bits=(2, 4), p=1),
                # AutoContrast (CHANGED to CLAHE)
                albu.CLAHE(p=1),
                # Equalize
                albu.Equalize(p=1, mode='pil'),
            ], p=1),
            ### rrc
            albu_fork.WindowResize(min_size=min_size, max_size=max_size),
            albu.PadIfNeeded(min_width=crop_res, min_height=crop_res, border_mode=cv2.BORDER_CONSTANT, value=0.0),
            albu.OneOf([
                albu_fork.RandomFocusedCrop(height=crop_res, width=crop_res, p=rfc_prob),
                albu.RandomSizedBBoxSafeCrop(crop_res, crop_res, erosion_rate=rbsc_er, interpolation=1, p=rsc_prob),
            ], p=1),
            albu.HorizontalFlip(p=0.5),
        ]
    albu_transform = albu.Compose(transform_list,
        bbox_params=albu.BboxParams(
            format='coco', label_fields=['category_ids', 'person_id', 'id', 'iou_thresh', 'is_known'],
            min_visibility=0.4,
        )
    )    
    albu_transform_dict = {'test': albu_transform}
    return AlbuWrapper(albu_transform, stat_dict)

