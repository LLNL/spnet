# Global imports
import os
import json
import collections
import numpy as np
import torch
import torchvision
### SSL imports
import math
import tqdm
from PIL import Image
from scipy.stats import loguniform
from torchvision.ops import boxes as box_ops
from torch.utils.data.sampler import BatchSampler, Sampler

# Utility function to get chunks
def chunks(l, n):
    """Yield successive n-sized chunks from list l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

# Dict of paths
ANNO_PATHS = {
    'trainval': 'coco/trainval.json',
    'test': 'coco/test.json',
    'train': 'coco/train.json',
    'val': 'coco/val.json',
    'minitrain': 'coco/minitrain.json',
    'minival': 'coco/minival.json',
    ###
    'train_p180': 'coco/train_p180.json',
    'train_p160': 'coco/train_p160.json',
    'train_p140': 'coco/train_p140.json',
    'train_p120': 'coco/train_p120.json',
    'train_p100': 'coco/train_p100.json',
    'train_p80': 'coco/train_p80.json',
    'train_p60': 'coco/train_p60.json',
    'train_p40': 'coco/train_p40.json',
    'train_p20': 'coco/train_p20.json',
}

IMAGE_PATHS = {
    'cuhk': 'Image/SSM',
    'prw': 'frames',
    'coco': 'train2017',
}

# Get labels and lookups
def get_coco_labels(root, dataset_name, image_set):
    # Image folder depends on dataset
    img_folder = IMAGE_PATHS[dataset_name]

    # Get anno file
    ann_file = ANNO_PATHS[image_set]
    img_folder = os.path.join(root, img_folder)
    ann_file = os.path.join(root, ann_file)

    # Load annotation file
    with open(ann_file, 'r') as fp:
        ann_dict = json.load(fp)

    # Load images
    img_set = set()
    ar_dict = {}
    for i, img in enumerate(ann_dict['images']):
        img_set.add(img['id'])
        w, h = img['width'], img['height']
        ar = w / h
        if ar <= 1:
            ar_dict[img['id']] = 0
        else:
            ar_dict[img['id']] = 1
    img_list = sorted(list(img_set))

    # Load annos
    pid_set, uid_set, aid_set = set(), set(), set()
    pid_img_set_dict = collections.defaultdict(set)
    for i, ann in enumerate(ann_dict['annotations']):
        puid = ann['person_id']
        is_known = ann['is_known']
        image_id = ann['image_id']
        pid_img_set_dict[puid].add(image_id)
        aid_set.add(puid)
        if is_known:
            pid_set.add(puid)
        else:
            uid_set.add(puid)
    pid_list, uid_list = sorted(list(pid_set)), sorted(list(uid_set))
    pid_lookup_dict = {
        'pid_lookup': {},
        'label_lookup': {},
        'num_img': len(img_list),
        'num_pid': len(pid_list),
        'num_uid': len(uid_list),
    }

    # Create lookup for labels to be used for OIM loss
    for idx, pid in enumerate(pid_list, 1):
        pid_lookup_dict['label_lookup'][pid] = idx

    # Create lookup with unique labels for each pid, to be used for other losses
    aid_list = sorted(list(aid_set))
    for idx, aid in enumerate(aid_list):
        pid_lookup_dict['pid_lookup'][aid] = idx

    # Return outputs
    return pid_lookup_dict, ar_dict

# Get datasets
def get_coco_dataset(root, dataset_name, image_set, transforms,
        pid_lookup_dict,
        ssl=False, ssl_anno_params={}):

    # Image folder depends on dataset
    img_folder = IMAGE_PATHS[dataset_name]

    # Get anno file
    ann_file = ANNO_PATHS[image_set]
    img_folder = os.path.join(root, img_folder)
    ann_file = os.path.join(root, ann_file)

    # Load special dataset loading transform
    t = [ConvertCoco(pid_lookup_dict)]

    if transforms is not None:
        t.append(transforms)
    transforms = torchvision.transforms.Compose(t)

    # Switch to load SSL vs. regular dataset
    if ssl:
        dataset = CocoSSLDetection(img_folder, ann_file, transforms=transforms,
            ssl_anno_params=ssl_anno_params)
    else:
        dataset = CocoDetection(img_folder, ann_file, transforms=transforms)

    # Hacky: remove images without annotations if partition has 'train' string in it
    if (dataset_name != 'in1k') and ('train' in image_set):
        len_before = len(dataset)
        dataset = _remove_images_without_annotations(dataset)
        len_after = len(dataset)
        print('==> Removed images in "{}" without annotations: {}/{}'.format(
            image_set, len_before - len_after, len_before))

    # De-nest dataset if needed
    if hasattr(dataset, 'dataset'):
        print('==> De-nesting dataset')
        dataset = dataset.dataset

    # Return outputs
    return dataset


# SSL Sampler
class SSLBatchSampler(BatchSampler):

    def __init__(self, sampler, num_images, num_views,
        num_replicas=1, rank=None, shuffle=True, seed=0):
        if not isinstance(sampler, Sampler):
            raise ValueError(
                "sampler should be an instance of "
                "torch.utils.data.Sampler, but got sampler={}".format(sampler)
            )

        # Distributed params
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0
        self.drop_last = False

        # Params
        self.sampler = sampler
        self.num_images = num_images
        self.num_views = num_views
        self.batch_size = self.num_images * self.num_views

        # Instantiate replica list
        self.replica_list = iter([])

    def __iter__(self):
        return iter(self.replica_list)

    def __len__(self):
        return int(math.ceil(((len(self.sampler) * self.num_views) / self.batch_size) / self.num_replicas))

    def set_epoch(self, epoch):
        print('*** CALLING SET EPOCH***')
        ### Set the epoch
        self.epoch = epoch

        ### Prep the batch and determine its length
        # Set random seed
        np.random.seed(self.seed + self.epoch)

        # Shuffle the index list
        sampler_idx_list = list(self.sampler)
        np.random.shuffle(sampler_idx_list)

        # Get indices for single view of each image for this process
        _replica_list = sampler_idx_list[self.rank::self.num_replicas]

        # Duplicate indices num_views times (sequentially i.e. [1, 1, 2, 2, ...] etc.)
        _dup_replica_list = np.repeat(_replica_list, self.num_views).tolist()

        # Split replicas into batches
        self.replica_list = list(chunks(_dup_replica_list, self.batch_size))

        # Return unduplicated replicate list for annotation generation
        return _replica_list


class TestSampler(torch.utils.data.Sampler):
    def __init__(self, partition_name, dataset, retrieval_dir, retrieval_name_list):
        # If dataset is subset, get dataset object
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
        # Store params
        self.partition_name = partition_name
        self.dataset = dataset
        self.retrieval_dir = retrieval_dir
        self.retrieval_name_list = retrieval_name_list
        image_id_set = set()
        if 'all' in retrieval_name_list:
            # List of all image_id that we need to gather detects and/or GT features for
            self.image_idx_list = range(len(self.dataset))
            # List of all anno id that we need to gather GT features for
            self.query_id_list = [int(x) for x in list(dataset.coco.anns.keys())]
        else:
            query_id_set, image_id_set = set(), set()
            for retrieval_name in retrieval_name_list:
                retrieval_path = os.path.join(retrieval_dir, '{}.json'.format(retrieval_name))
                with open(retrieval_path, 'r') as fp:
                    retrieval_dict = json.load(fp)
                    _image_id_set = set(retrieval_dict['images'])
                    image_id_set = image_id_set.union(_image_id_set) 
                    # NOTE: retrieval_dict['queries'] can be either a dict or a list, but this is correct in either case
                    _query_id_set = set(retrieval_dict['queries'])
                    query_id_set = query_id_set.union(_query_id_set)
                    print(retrieval_name, len(_image_id_set), len(image_id_set))
                    # The retrieval dict can be large, so delete it to free up space
                    del retrieval_dict
            # List of all image_id that we need to gather detects and/or GT features for
            image_id_list = list(image_id_set)
            self.image_idx_list = [dataset.ids.index(_id) for _id in image_id_list]
            self.query_id_list = [int(x) for x in list(query_id_set)]

    def __iter__(self):
        for image_idx in self.image_idx_list:
            yield image_idx

    def __len__(self):
        return len(self.image_idx_list)


#
class ConvertCoco(object):
    def __init__(self, label_lookup_dict):
        self.label_lookup_dict = label_lookup_dict

    def __call__(self, data):
        image, target = data
        w, h = image.size

        image_id = target['image_id']
        image_id = torch.tensor([image_id])

        if target['image_label'] is not None:
            image_label = target['image_label']
            image_label = torch.tensor([image_label])
            use_image_label = True
        else:
            use_image_label = False

        anno = target['annotations']

        boxes = [obj['bbox'] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        # Convert data to tensors
        classes = [obj['category_id'] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)
        ids = [obj['id'] for obj in anno]
        ids = torch.tensor(ids, dtype=torch.int64)
        iou_thresh = [obj['iou_thresh'] for obj in anno]
        iou_thresh = torch.tensor(iou_thresh, dtype=torch.float32)
        is_known = [obj['is_known'] for obj in anno]
        is_known = torch.tensor(is_known, dtype=torch.bool)

        # Build mask of valid bboxes to keep
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]

        # Convert boxes to x, y, w, h
        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

        # Apply valid bbox keep mask to all data
        classes = classes[keep]
        ids = ids[keep]
        iou_thresh = iou_thresh[keep]
        is_known = is_known[keep]

        # Create target dict
        target = {}
        target['boxes'] = boxes
        target['image_id'] = image_id
        if use_image_label:
            target['image_label'] = image_label

        # for conversion to coco api
        area = torch.tensor([obj['area'] for obj in anno])

        # Keep only valid PIDs
        try:
            person_id_arr = np.array([obj['person_id'] for obj in anno], dtype=object)[keep.numpy()]
        except IndexError:
            raise Exception({'keep': keep})

        # Store info in target dict
        target['area'] = area
        target['person_id'] = person_id_arr.tolist()
        target['image_size'] = torch.FloatTensor([w, h])
        target['id'] = ids.tolist()
        target['iou_thresh'] = iou_thresh
        target['is_known'] = is_known

        # Store OIM label
        if self.label_lookup_dict is not None:
            target['labels'] = torch.LongTensor([self.label_lookup_dict['label_lookup'][pid] if pid in self.label_lookup_dict['label_lookup'] else self.label_lookup_dict['num_pid']+1 for pid in target['person_id']])
        else:
            target['labels'] = classes

        # Store PID label
        if self.label_lookup_dict is not None:
            target['person_id'] = [self.label_lookup_dict['pid_lookup'][pid] for pid in target['person_id']]

        # Check lens are valid
        assert len(target['boxes']) == len(target['labels']) == len(target['person_id'])

        # Check everything for correct dimensions
        assert boxes.size(0) == classes.size(0) == person_id_arr.shape[0]

        return (image, target)


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms

    def __getitem__(self, idx):
        # Get Image ID
        image_id = self.ids[idx]
        # Get image label if it is available
        if 'label' in self.coco.imgs[image_id]:
            image_label = self.coco.imgs[image_id]['label']
        else:
            image_label = None 
        # Load image, target
        img, _target = super(CocoDetection, self).__getitem__(idx)
        target = dict(image_id=image_id, image_label=image_label, annotations=_target)
        # Make sure transforms are available
        if self._transforms is not None:
            data = (img, target)
            out = self._transforms(data)
            return out
        else:
            raise Exception

@torch.no_grad()
def _get_box_var(boxes, iou_range, debug=False):
    boxes = torch.tensor(boxes)
    # Convert boxes to x1y1x2y2
    boxes[:, 2] += boxes[:, 0]
    boxes[:, 3] += boxes[:, 1]
    # Number of boxes
    n = boxes.size(0)
    # Number of variations for each different positive type
    k = 2
    # Get box width, height
    boxes_wh = boxes[:, 2:] - boxes[:, :2]
    boxes_ar = boxes_wh[:, 0] / boxes_wh[:, 1]
    boxes_cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
    boxes_cy = (boxes[:, 1] + boxes[:, 3]) / 2.0

    # Build IoU target
    if min(iou_range) == max(iou_range):
        iou_target = torch.ones(n, 6).to(boxes.device) * iou_range[0]
    else:
        ## Generate random IoU target
        _iou_target = torch.rand(n, 6).to(boxes.device)
        ## Normalize IoU target to iou_range
        _iou_target = _iou_target * (max(iou_range) - min(iou_range))
        iou_target = _iou_target + min(iou_range)

    # Fixed height
    ## Fat
    w_fat = boxes_wh[:, 0] / iou_target[:, 0]
    ## Skinny
    w_skinny = boxes_wh[:, 0] * iou_target [:, 1]
    wfh = torch.stack([w_fat, w_skinny]) / 2.0
    ## Build fixed-height boxes
    fh_boxes_y1 = boxes[:, 1].reshape(1, n).repeat(k, 1).permute(1, 0).unsqueeze(2)
    fh_boxes_y2 = boxes[:, 3].reshape(1, n).repeat(k, 1).permute(1, 0).unsqueeze(2)
    ### Fixed-height positive
    fh_boxes_x1 = (boxes_cx - wfh).permute(1, 0).unsqueeze(2)
    fh_boxes_x2 = (boxes_cx + wfh).permute(1, 0).unsqueeze(2)
    fh_boxes = torch.cat([fh_boxes_x1, fh_boxes_y1, fh_boxes_x2, fh_boxes_y2], dim=2)

    # Fixed width
    ## Tall
    h_tall = boxes_wh[:, 1] / iou_target[:, 2]
    ## Short
    h_short = boxes_wh[:, 1] * iou_target[:, 3]
    hfw = torch.stack([h_tall, h_short]) / 2.0
    ## Build fixed-height boxes
    fw_boxes_x1 = boxes[:, 0].reshape(1, n).repeat(k, 1).permute(1, 0).unsqueeze(2)
    fw_boxes_x2 = boxes[:, 2].reshape(1, n).repeat(k, 1).permute(1, 0).unsqueeze(2)
    ### Fixed-height positive
    fw_boxes_y1 = (boxes_cy - hfw).permute(1, 0).unsqueeze(2)
    fw_boxes_y2 = (boxes_cy + hfw).permute(1, 0).unsqueeze(2)
    fw_boxes = torch.cat([fw_boxes_x1, fw_boxes_y1, fw_boxes_x2, fw_boxes_y2], dim=2)

    # Fixed aspect ratio
    ## Big
    h_big = boxes_wh[:, 1] / torch.sqrt(iou_target[:, 4])
    w_big = h_big * boxes_ar
    ## Small
    h_small = boxes_wh[:, 1] * torch.sqrt(iou_target[:, 5])
    w_small = h_small * boxes_ar
    ## Build fixed-AR boxes
    wfar = torch.stack([w_big, w_small]) / 2.0
    hfar = torch.stack([h_big, h_small]) / 2.0
    ### Fixed-AR positive
    far_boxes_x1 = (boxes_cx - wfar).permute(1, 0).unsqueeze(2)
    far_boxes_x2 = (boxes_cx + wfar).permute(1, 0).unsqueeze(2)
    far_boxes_y1 = (boxes_cy - hfar).permute(1, 0).unsqueeze(2)
    far_boxes_y2 = (boxes_cy + hfar).permute(1, 0).unsqueeze(2)
    far_boxes = torch.cat([far_boxes_x1, far_boxes_y1, far_boxes_x2, far_boxes_y2], dim=2)

    # Stack positive and negative boxes
    v_boxes = torch.cat([fh_boxes, fw_boxes, far_boxes], dim=1)

    # Check all boxes that IoU is correct within tol
    if debug:
        iou = box_ops.box_iou_ew(boxes, v_boxes)
        assert torch.allclose(iou, iou_target)

    # Pick 1
    overlap_idx = torch.randint(0, 6, (n,)).view(n, 1, 1).repeat(1, 1, 4)
    v_boxes = torch.gather(v_boxes, 1, overlap_idx).squeeze(1)

    # Convert boxes to xywh
    v_boxes[:, 2] -= v_boxes[:, 0]
    v_boxes[:, 3] -= v_boxes[:, 1]

    # Back to numpy
    v_boxes = v_boxes.numpy()

    # Return v_boxes
    return v_boxes

class BoxGenerator:
    def __init__(self, img_folder, ids, imgs, rank=0, num_anno=100,
            min_width=8, max_width=128, min_ar=1.0/3.0, max_ar=3.0,
            iou_thresh=0.1, anno_method='basic',
            filter_monotone=False, filter_duplicate=False):
        # Data
        self.img_folder = img_folder
        self.ids = ids
        self.imgs = imgs

        # Distributed
        self.rank = rank

        # Anno params
        ## Num anno params
        self.num_anno = num_anno
        print('==> Num anno per image: {}'.format(self.num_anno))
        ## Width params
        self.min_width = min_width
        self.max_width = max_width
        ## AR params
        self.min_ar = min_ar
        self.max_ar = max_ar
        ## Thresh for anno_method == 'iou'
        self.iou_thresh = iou_thresh
        ## Anno method \in {'basic', 'iou'}
        self.anno_method = anno_method
        ## Filtering options
        self.filter_monotone = filter_monotone
        self.filter_duplicate = filter_duplicate
        self.filter = self.filter_monotone or self.filter_duplicate
        ##
        self.filter_patch_size = (8, 8)
        self.monotone_thresh = 16
        self.duplicate_thresh = 512

    def generate_anno(self, index_list, seed, epoch):
        # Generate annotations
        if self.anno_method == 'basic':
            return self.generate_anno_basic(index_list, seed, epoch)
        elif self.anno_method == 'iou':
            return self.generate_anno_iou(index_list, seed, epoch)
        elif self.anno_method == 'overlap':
            return self.generate_anno_overlap(index_list, seed, epoch)

    def generate_anno_iou(self, index_list, seed, epoch):
        ### Prep the batch and determine its length
        # Set random seed
        np.random.seed(seed)

        # Convert ids and indices to numpy array for fast indexing
        id_arr = np.array(self.ids)
        index_arr = np.array(index_list)

        ### Set anns
        anno_dict = {}
        anno_list = []
        anno_id = (self.rank * len(self.ids) * self.num_anno) + (epoch * len(self.ids) * self.num_anno)
        print('Epoch, anno_id:', epoch, anno_id)
        monotone_counter, duplicate_counter = 0, 0
        codebook = []
        for image_id in tqdm.tqdm(id_arr[index_arr]):
            image_dict = self.imgs[image_id]
            if self.filter:
                image_file = image_dict['file_name']
                image_path = os.path.join(self.img_folder, image_file)
                image = Image.open(image_path).convert('L')
            image_width = image_dict['width']
            image_height = image_dict['height']
            anno_dict[image_id] = []
            while len(anno_dict[image_id]) < self.num_anno:
                known_box_list = [t['bbox'] for t in anno_dict[image_id]]
                if len(known_box_list) > 0:
                    _known_box_tsr = torch.FloatTensor(known_box_list)
                    known_box_tsr = _known_box_tsr.clone()
                    known_box_tsr[:, 2:] += known_box_tsr[:, :2]
                # Clip max width
                max_width = min(self.max_width, image_width)
                if self.min_width == max_width:
                    anno_width = self.min_width
                else:
                    anno_width = math.ceil(loguniform.rvs(self.min_width, max_width, size=1).item())
                # XXX: loguniform instead?
                anno_ar = loguniform.rvs(self.min_ar, self.max_ar, size=1).item()
                # Compute height
                anno_height = math.ceil(anno_width / anno_ar)
                # Clip anno height
                anno_height = min(anno_height, image_height)
                anno_area = float(anno_width * anno_height)
                # Set anno x position
                if image_width == anno_width:
                    anno_x = 0
                else:
                    anno_x = np.random.randint(0, image_width - anno_width)
                # Set anno y position
                if image_height == anno_height:
                    anno_y = 0
                else:
                    anno_y = np.random.randint(0, image_height - anno_height)
                # Set anno bbox
                anno_bbox = [float(anno_x), float(anno_y), float(anno_width), float(anno_height)]
                # Assert that the bbox is within bounds of image
                assert 0.0 <= anno_bbox[0] < image_width
                assert 0.0 <= anno_bbox[1] < image_height
                assert 0.0 < anno_bbox[0] + anno_bbox[2] <= image_width
                assert 0.0 < anno_bbox[1] + anno_bbox[3] <= image_height

                # Check if it overlaps with existing boxes
                if len(known_box_list) > 0:
                    anno_box_tsr = torch.FloatTensor(anno_bbox).unsqueeze(0)
                    anno_box_tsr[2:] += anno_box_tsr[:2]
                    iou_tsr = box_ops.box_iou(known_box_tsr, anno_box_tsr)

                # Compute the image code
                if self.filter:
                    x1, y1, w, h = anno_x, anno_y, anno_width, anno_height
                    x2, y2 = x1 + w, y1 + h
                    image_patch = image.crop((x1, y1, x2, y2))
                    image_code = np.array(image_patch.resize(self.filter_patch_size)).flatten().reshape(1, -1)
                    ## Filter monotone codes
                    if self.filter_monotone:
                        image_range = image_code.max() - image_code.min()
                        if image_range < self.monotone_thresh:
                            monotone_counter += 1
                            continue

                    ## Filter duplicate codes
                    if self.filter_duplicate:
                        if len(codebook) > 0:
                            ### Check against codebook
                            diff = np.abs(codebook - image_code).sum(axis=1)
                            if diff.min() < self.duplicate_thresh:
                                duplicate_counter += 1
                                continue
                        else:
                            ### Add first code
                            codebook = image_code
                
                # Store the annotations
                if (len(known_box_list) == 0) or torch.all(iou_tsr < self.iou_thresh):
                    # Build anno dict
                    anno_dict[image_id].append({
                        'area': anno_area,
                        'bbox': anno_bbox,
                        'category_id': 1,
                        'id': anno_id,
                        'image_id': image_id.item(),
                        'iscrowd': 0,
                        'person_id': anno_id,#'p{}'.format(anno_id),
                        'iou_thresh': 0.5,
                        'is_known': True,
                    })
                    # Increment anno_id
                    anno_list.append(anno_id)
                    anno_id += 1
                    ### Add new code to codebook
                    if self.filter_duplicate and ((anno_id % 10) == 0):
                        codebook = np.append(codebook, image_code, axis=0)

        print('Num monotone filtered: {}/{}'.format(monotone_counter, anno_id))
        print('Num duplicate filtered: {}/{}'.format(duplicate_counter, anno_id))
        return anno_list, anno_dict

    def generate_anno_basic(self, index_list, seed, epoch):
        #print(f'rank {self.rank} indices: min={min(index_list)}, max={max(index_list)}')
        ### Prep the batch and determine its length
        # Set random seed
        np.random.seed(seed)

        # Convert ids and indices to numpy array for fast indexing
        id_arr = np.array(self.ids)
        index_arr = np.array(index_list)

        ### Set anns
        anno_dict = {}
        anno_list = []
        anno_id = (self.rank * len(self.ids) * self.num_anno) + (epoch * len(self.ids) * self.num_anno)
        print('Epoch, anno_id:', epoch, anno_id)
        #for image_id in tqdm.tqdm(self.ids):
        for image_id in tqdm.tqdm(id_arr[index_arr]):
            image_dict = self.imgs[image_id]
            image_width = image_dict['width']
            image_height = image_dict['height']
            anno_dict[image_id] = []

            max_width = min(self.max_width, image_width)
            min_width = min(self.min_width, max_width-1)
            assert 0 < min_width < max_width, f'0 < {min_width} < {max_width}: False. self.max_width={self.max_width}, image_width={image_width}'
            anno_width = np.ceil(loguniform.rvs(min_width, max_width, size=self.num_anno))
            # XXX: loguniform instead?
            anno_ar = loguniform.rvs(self.min_ar, self.max_ar, size=self.num_anno)
            # Compute height
            anno_height = np.ceil(anno_width / anno_ar)
            # Clip anno height
            anno_height = np.minimum(anno_height, image_height)
            anno_area = anno_width * anno_height
            # Set anno x position
            anno_x_f = np.random.rand(self.num_anno)
            anno_x = anno_x_f * (image_width - anno_width)
            anno_x = anno_x.clip(min=0)
            # Set anno y position
            anno_y_f = np.random.rand(self.num_anno)
            anno_y = anno_y_f * (image_height - anno_height)
            anno_y = anno_y.clip(min=0)
            # Set anno bbox
            anno_bbox = np.vstack([anno_x, anno_y, anno_width, anno_height]).T
            # Assert that the bbox is within bounds of image
            assert np.all(0.0 <= anno_bbox[:, 0]) and np.all(anno_bbox[:, 0] < image_width)
            assert np.all(0.0 <= anno_bbox[:, 1]) and np.all(anno_bbox[:, 1] < image_height)
            assert np.all(0.0 < (anno_bbox[:, 0] + anno_bbox[:, 2]))
            assert np.all((anno_bbox[:, 0] + anno_bbox[:, 2]) <= image_width)
            assert np.all(0.0 < (anno_bbox[:, 1] + anno_bbox[:, 3]))
            assert np.all((anno_bbox[:, 1] + anno_bbox[:, 3]) <= image_height)

            # Build anno dict
            for _anno_bbox, _anno_area in zip(anno_bbox.tolist(), anno_area.tolist()):
                anno_dict[image_id].append({
                    'area': _anno_area,
                    'bbox': _anno_bbox,
                    'category_id': 1,
                    'id': anno_id,
                    'image_id': image_id.item(),
                    'iscrowd': 0,
                    'person_id': anno_id,#'p{}'.format(anno_id),
                    'iou_thresh': 0.5,
                    'is_known': True,
                })
                # Increment anno_id
                anno_list.append(anno_id)
                anno_id += 1

        return anno_list, anno_dict

    def generate_anno_overlap(self, index_list, seed, epoch):
        #print(f'rank {self.rank} indices: min={min(index_list)}, max={max(index_list)}')
        ### Prep the batch and determine its length
        # Set random seed
        np.random.seed(seed)

        # Convert ids and indices to numpy array for fast indexing
        id_arr = np.array(self.ids)
        index_arr = np.array(index_list)

        ### Set anns
        anno_dict = {}
        anno_list = []
        anno_id = self.rank * len(self.ids) * self.num_anno // 2
        #for image_id in tqdm.tqdm(self.ids):
        for image_id in tqdm.tqdm(id_arr[index_arr]):
            image_dict = self.imgs[image_id]
            image_width = image_dict['width']
            image_height = image_dict['height']
            anno_dict[image_id] = []

            max_width = min(self.max_width, image_width)
            min_width = min(self.min_width, max_width-1)
            assert 0 < min_width < max_width, f'0 < {min_width} < {max_width}: False. self.max_width={self.max_width}, image_width={image_width}'
            anno_width = np.ceil(loguniform.rvs(min_width, max_width, size=self.num_anno//2))
            # XXX: loguniform instead?
            anno_ar = loguniform.rvs(self.min_ar, self.max_ar, size=self.num_anno//2)
            # Compute height
            anno_height = np.ceil(anno_width / anno_ar)
            # Clip anno height
            anno_height = np.minimum(anno_height, image_height)
            # Set anno x position
            anno_x_f = np.random.rand(self.num_anno//2)
            anno_x = anno_x_f * (image_width - anno_width)
            anno_x = anno_x.clip(min=0)
            # Set anno y position
            anno_y_f = np.random.rand(self.num_anno//2)
            anno_y = anno_y_f * (image_height - anno_height)
            anno_y = anno_y.clip(min=0)
            # Set anno bbox
            anno_bbox = np.vstack([anno_x, anno_y, anno_width, anno_height]).T
            # Make more annos based on these
            overlap_bbox = _get_box_var(anno_bbox, iou_range=(0.6, 0.6))
            ## Clip new boxes: WARNING: will violate IoU constraint after clipping
            overlap_bbox[:, 0] = overlap_bbox[:, 0].clip(min=0.0, max=image_width-1)
            overlap_bbox[:, 1] = overlap_bbox[:, 1].clip(min=0.0, max=image_height-1)
            overlap_bbox[:, 2] = overlap_bbox[:, 2].clip(min=1, max=image_width-overlap_bbox[:, 0])
            overlap_bbox[:, 3] = overlap_bbox[:, 3].clip(min=1, max=image_height-overlap_bbox[:, 1])
            #
            anno_bbox = np.concatenate([anno_bbox, overlap_bbox])
            # Compute area
            anno_area = anno_bbox[:, 2] * anno_bbox[:, 3]
            # Assert that the bbox is within bounds of image
            assert np.all(anno_area > 0.0)
            assert np.all(0.0 <= anno_bbox[:, 0]) and np.all(anno_bbox[:, 0] < image_width)
            assert np.all(0.0 <= anno_bbox[:, 1]) and np.all(anno_bbox[:, 1] < image_height)
            assert np.all(0.0 < (anno_bbox[:, 0] + anno_bbox[:, 2]))
            assert np.all((anno_bbox[:, 0] + anno_bbox[:, 2]) <= image_width)
            assert np.all(0.0 < (anno_bbox[:, 1] + anno_bbox[:, 3]))
            assert np.all((anno_bbox[:, 1] + anno_bbox[:, 3]) <= image_height)

            # Build anno dict
            for _anno_bbox, _anno_area in zip(anno_bbox.tolist(), anno_area.tolist()):
                anno_dict[image_id].append({
                    'area': _anno_area,
                    'bbox': _anno_bbox,
                    'category_id': 1,
                    'id': anno_id,
                    'image_id': image_id.item(),
                    'iscrowd': 0,
                    'person_id': anno_id,#'p{}'.format(anno_id),
                    'iou_thresh': 0.5,
                    'is_known': True,
                })
                # Increment anno_id
                anno_list.append(anno_id)
                anno_id += 1

        return anno_list, anno_dict


class CocoSSLDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, ssl_anno_params={}, seed=0):
        super(CocoSSLDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.id_batch_size_dict = None
        self.seed = seed

        # Set rank
        try:
            self.rank = torch.distributed.get_rank()
        except RuntimeError:
            self.rank = 0

        # Insantiate index list
        self.valid_index_list = []

        # Instantiate box generator
        self.box_generator = BoxGenerator(img_folder, self.ids, self.coco.imgs, rank=self.rank,
            **ssl_anno_params)

    def generate_anno(self, index_list):
        seed = self.seed + self.epoch
        self.anno_list, self.anno_dict = self.box_generator.generate_anno(
            index_list, seed, self.epoch)

    def set_epoch(self, epoch, index_list):
        ### Set the epoch
        self.epoch = epoch
        # Generate annotations for this epoch
        self.generate_anno(index_list)
        # Set valid index list
        self.valid_index_list = index_list
        print(f'SET EPOCH | epoch: {epoch}, rank: {self.rank}, index_list={index_list[:10]}')

    def _new_load_target(self, id):
        return self.anno_dict[id]

    def _load_image(self, id):
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def __getitem__(self, idx):
        # Get Image ID
        assert idx in self.valid_index_list, f'FAIL | epoch: {self.epoch}, rank: {self.rank}, idx={idx}, index_list={self.valid_index_list[:10]}'
        image_id = self.ids[idx]
        # Get image label if it is available
        if 'label' in self.coco.imgs[image_id]:
            image_label = self.coco.imgs[image_id]['label']
        else:
            image_label = None 
        # Get image
        img = self._load_image(image_id)
        # Build target
        _target = self._new_load_target(image_id)
        target = dict(image_id=image_id, image_label=image_label, annotations=_target)
        # Make sure transforms are available
        if self._transforms is not None:
            data = (img, target)
            out = self._transforms(data)
            return out
        else:
            raise Exception


def _remove_images_without_annotations(dataset, cat_list=None):
    def _has_only_empty_bbox(anno):
        return all(any(o <= 1 for o in obj['bbox'][2:]) for obj in anno)

    def _has_valid_annotation(anno):
        # if it's empty, there is no annotation
        if len(anno) == 0:
            return False
        # if all boxes have close to zero area, there is no annotation
        if _has_only_empty_bbox(anno):
            return False
        else:
            return True

    assert isinstance(dataset, torchvision.datasets.CocoDetection)
    ids = []
    for ds_idx, img_id in enumerate(dataset.ids):
        ann_ids = dataset.coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anno = dataset.coco.loadAnns(ann_ids)
        if cat_list:
            anno = [obj for obj in anno if obj['category_id'] in cat_list]
        if _has_valid_annotation(anno):
            ids.append(ds_idx)

    dataset = torch.utils.data.Subset(dataset, ids)
    return dataset
