# Global imports
import os
import json
import collections
import copy
import tqdm
import numpy as np
from PIL import Image
import concurrent.futures


## Helper functions for cocop_sample_ablation
def _sample(candidate_list, s, t, eps, frac=0.8):
    # Calculate how many samples to keep
    num_keep = int(np.ceil(frac * t)) - s
    print(num_keep, len(candidate_list))
    # Shuffle candidate list
    np.random.shuffle(candidate_list)
    # Subsample list
    keep_list = candidate_list[:num_keep]
    actual_frac = (len(keep_list) + s) / t
    # Make sure final frac is close to target
    print(actual_frac, frac)
    assert np.abs(frac - actual_frac) < eps
    # Return keep list
    return keep_list

def _get_anno_dict(anno_dict, candidate_list, singleton_list):
    new_anno_dict = collections.defaultdict(list)
    for image_id, anno_idx in singleton_list:
        new_anno_dict[image_id].append(anno_dict[image_id][anno_idx])
    for image_id, anno_idx in candidate_list:
        new_anno_dict[image_id].append(anno_dict[image_id][anno_idx])
    return dict(new_anno_dict)

def _check_anno_dict(anno_dict, t, eps, frac=0.8):
    c = 0
    for k, v in anno_dict.items():
        assert len(v) >= 1
        c += len(v)
    actual_frac = c / t
    # Make sure final frac is close to target
    print(actual_frac, frac)
    assert np.abs(frac - actual_frac) < eps

def _convert_coco(anno_dict, coco_lookup, coco_dict, empty_coco_dict):
    new_coco_dict = copy.deepcopy(empty_coco_dict)
    for image_id, anno_list in anno_dict.items():
        image_idx = coco_lookup[image_id]
        new_coco_dict['images'].append(coco_dict['images'][image_idx])
        new_coco_dict['annotations'].extend(anno_list)
    return new_coco_dict

def _save_anno_dict(anno_dict, anno_name, coco_dir):
    anno_path = os.path.join(coco_dir, f'{anno_name}.json')
    print(f'Saving to: {anno_path}')
    with open(anno_path, 'w') as fp:
        json.dump(anno_dict, fp)


## Sample ablation partitioning function
def cocop_sample_ablation(coco_dir, seed=0):
    """Sampling algorithm:
    Goal: Leave at least one person box in every image.
    Each smaller subset should be a subset of each larger subset.
    1) Randomly mark one anno in each image that cannot be removed.
    2) Randomly subsample from remaining unmarked annos to reach exactly desired fraction.
    3) Repeat sequentially for each partition using previous partition.
    """
    # Seed for reproducibility
    np.random.seed(seed)

    # Set up directories
    anno_dir = f'{coco_dir}/coco'
    coco_path = f'{anno_dir}/train.json'

    # Get coco dict for train partition
    with open(coco_path, 'r') as fp:
        coco_dict = json.load(fp)

    empty_coco_dict = copy.deepcopy(coco_dict)
    empty_coco_dict['images'] = []
    empty_coco_dict['annotations'] = []

    coco_lookup = {}
    for image_idx, image in enumerate(coco_dict['images']):
        image_id = image['id']
        coco_lookup[image_id] = image_idx

    anno_dict = collections.defaultdict(list)
    print(len(coco_dict['annotations']))
    for anno in coco_dict['annotations']:
        anno_dict[anno['image_id']].append(anno)
    anno_dict = dict(anno_dict)

    print(len(anno_dict))
    anno_dict_p100 = copy.deepcopy(anno_dict)
    for image_id in list(anno_dict_p100.keys()):
        if len(anno_dict_p100[image_id]) < 2:
            del anno_dict_p100[image_id]
    print(len(anno_dict_p100))

    s, t = 0, 0
    for k, v in anno_dict_p100.items():
        s += 1
        t += len(v)
    p = s / t
    print('Frac singleton:', p)

    singleton_list = []
    candidate_list_p100 = []
    for image_id, anno_list in tqdm.tqdm(anno_dict_p100.items()):
        # 1) Randomly mark one anno to keep in each image
        n = len(anno_list)
        i = np.random.randint(0, n)
        singleton_list.append((image_id, i))
        # Mark all other annos as candidates
        for j in range(n):
            if i != j:
                candidate_list_p100.append((image_id, j))

    eps = 2 / t
    candidate_list_p80 = _sample(candidate_list_p100, s, t, eps, frac=0.8)
    candidate_list_p60 = _sample(candidate_list_p80, s, t, eps, frac=0.6)
    candidate_list_p40 = _sample(candidate_list_p60, s, t, eps, frac=0.4)
    candidate_list_p20 = _sample(candidate_list_p40, s, t, eps, frac=0.2)

    # 100%
    _check_anno_dict(anno_dict_p100, t, eps, frac=1.0)
    coco_dict_p100 = _convert_coco(anno_dict_p100, coco_lookup, coco_dict, empty_coco_dict)
    _save_anno_dict(coco_dict_p100, 'train_p100', anno_dir)

    # 80%
    anno_dict_p80 = _get_anno_dict(anno_dict_p100, candidate_list_p80, singleton_list)
    _check_anno_dict(anno_dict_p80, t, eps, frac=0.8)
    coco_dict_p80 = _convert_coco(anno_dict_p80, coco_lookup, coco_dict, empty_coco_dict)
    _save_anno_dict(coco_dict_p80, 'train_p80', anno_dir)

    # 60%
    anno_dict_p60 = _get_anno_dict(anno_dict_p100, candidate_list_p60, singleton_list)
    _check_anno_dict(anno_dict_p60, t, eps, frac=0.6)
    coco_dict_p60 = _convert_coco(anno_dict_p60, coco_lookup, coco_dict, empty_coco_dict)
    _save_anno_dict(coco_dict_p60, 'train_p60', anno_dir)

    # 40%
    anno_dict_p40 = _get_anno_dict(anno_dict_p100, candidate_list_p40, singleton_list)
    _check_anno_dict(anno_dict_p40, t, eps, frac=0.4)
    coco_dict_p40 = _convert_coco(anno_dict_p40, coco_lookup, coco_dict, empty_coco_dict)
    _save_anno_dict(coco_dict_p40, 'train_p40', anno_dir)

    # 20%
    anno_dict_p20 = _get_anno_dict(anno_dict_p100, candidate_list_p20, singleton_list)
    _check_anno_dict(anno_dict_p20, t, eps, frac=0.2)
    coco_dict_p20 = _convert_coco(anno_dict_p20, coco_lookup, coco_dict, empty_coco_dict)
    _save_anno_dict(coco_dict_p20, 'train_p20', anno_dir)

## Sample ablation partitioning function
def cocop_add_ablation(coco_dir, seed=0):
    """Sampling algorithm:
    Goal: Leave at least one person box in every image.
    Each smaller subset should be a subset of each larger subset.
    1) Randomly mark one anno in each image that cannot be removed.
    2) Randomly subsample from remaining unmarked annos to reach exactly desired fraction.
    3) Repeat sequentially for each partition using previous partition.
    """
    # Seed for reproducibility
    np.random.seed(seed)

    # Set up directories
    anno_dir = f'{coco_dir}/coco'
    coco_path = f'{anno_dir}/train.json'

    # Get coco dict for train partition
    with open(coco_path, 'r') as fp:
        coco_dict = json.load(fp)

    empty_coco_dict = copy.deepcopy(coco_dict)
    empty_coco_dict['images'] = []
    empty_coco_dict['annotations'] = []

    coco_lookup = {}
    image_size_lookup = {}
    for image_idx, image in enumerate(tqdm.tqdm(coco_dict['images'])):
        image_id = image['id']
        coco_lookup[image_id] = image_idx
        w, h = image['width'], image['height']
        image_size_arr = np.array([w, h, w, h])
        image_size_lookup[image_id] = image_size_arr

    anno_dict = collections.defaultdict(list)
    bbox_list = []
    print(len(coco_dict['annotations']))
    for anno in tqdm.tqdm(coco_dict['annotations']):
        anno_dict[anno['image_id']].append(anno)
        image_size_arr = image_size_lookup[anno['image_id']]
        bbox_list.append(np.array(anno['bbox']) / image_size_arr)
    bbox_arr = np.array(bbox_list)
    anno_dict = dict(anno_dict)

    print(len(anno_dict))
    anno_dict_p100 = copy.deepcopy(anno_dict)
    num_anno = 0
    for image_id in tqdm.tqdm(list(anno_dict_p100.keys())):
        if len(anno_dict_p100[image_id]) < 2:
            del anno_dict_p100[image_id]
        else:
            num_anno += len(anno_dict_p100[image_id])
    print(len(anno_dict_p100))
    print('Tot num anno:', num_anno)
    n_add = int(0.2*(num_anno))
    print('Add 20%:', n_add)

    def _add(_anno_dict, bbox_arr, n_add, curr_num_anno):
        rand_image_ids = np.random.choice(list(_anno_dict.keys()), n_add)
        rand_bbox_idx = np.random.choice(range(len(bbox_arr)), n_add)
        rand_bboxes = bbox_arr[rand_bbox_idx]
        num_added = 0
        for image_id, bbox in tqdm.tqdm(list(zip(rand_image_ids, rand_bboxes))):
            image_size_arr = image_size_lookup[image_id]
            _anno = copy.deepcopy(_anno_dict[image_id][0])
            _anno['id'] = curr_num_anno + num_added
            _anno['person_id'] = curr_num_anno + num_added
            scaled_bbox = bbox * image_size_arr
            _anno['bbox'] = scaled_bbox.tolist()
            _anno['area'] = scaled_bbox[2] * scaled_bbox[3]
            _anno_dict[image_id].append(_anno)
            num_added += 1
        return num_added

    curr_num_anno = 300000
    anno_dict_p120 = copy.deepcopy(anno_dict_p100)
    curr_num_anno += _add(anno_dict_p120, bbox_arr, n_add, curr_num_anno)
    anno_dict_p140 = copy.deepcopy(anno_dict_p120)
    curr_num_anno += _add(anno_dict_p140, bbox_arr, n_add, curr_num_anno)
    anno_dict_p160 = copy.deepcopy(anno_dict_p140)
    curr_num_anno += _add(anno_dict_p160, bbox_arr, n_add, curr_num_anno)
    anno_dict_p180 = copy.deepcopy(anno_dict_p160)
    curr_num_anno += _add(anno_dict_p180, bbox_arr, n_add, curr_num_anno)
    print('Final num_anno:', curr_num_anno)

    # 120%
    t = num_anno
    eps = 2 / t
    _check_anno_dict(anno_dict_p120, t, eps, frac=1.2)
    coco_dict_p120 = _convert_coco(anno_dict_p120, coco_lookup, coco_dict, empty_coco_dict)
    _save_anno_dict(coco_dict_p120, 'train_p120', anno_dir)

    # 140%
    _check_anno_dict(anno_dict_p140, t, eps, frac=1.4)
    coco_dict_p140 = _convert_coco(anno_dict_p140, coco_lookup, coco_dict, empty_coco_dict)
    _save_anno_dict(coco_dict_p140, 'train_p140', anno_dir)

    # 160%
    _check_anno_dict(anno_dict_p160, t, eps, frac=1.6)
    coco_dict_p160 = _convert_coco(anno_dict_p160, coco_lookup, coco_dict, empty_coco_dict)
    _save_anno_dict(coco_dict_p160, 'train_p160', anno_dir)

    # 180%
    _check_anno_dict(anno_dict_p180, t, eps, frac=1.8)
    coco_dict_p180 = _convert_coco(anno_dict_p180, coco_lookup, coco_dict, empty_coco_dict)
    _save_anno_dict(coco_dict_p180, 'train_p180', anno_dir)


def coco2cocop(coco_dir):
    # Anno dir
    source_anno_dir = f'{coco_dir}/annotations'
    anno_dir = f'{coco_dir}/coco'

    # 
    if not os.path.exists(anno_dir):
        os.makedirs(anno_dir)

    # Train anno path
    train_anno_path = f'{source_anno_dir}/instances_train2017.json'
    new_train_anno_path = f'{anno_dir}/train.json'

    # Val anno path
    val_anno_path = f'{source_anno_dir}/instances_val2017.json'
    new_val_anno_path = f'{anno_dir}/val.json'

    with open(train_anno_path, 'r') as fp:
        train_anno_dict = json.load(fp)

    with open(val_anno_path, 'r') as fp:
        val_anno_dict = json.load(fp)

    def get_person_anno(anno_dict):
        # Get id for person category
        for cat in anno_dict['categories']:
            if cat['name'] == 'person':
                person_id = cat['id']
                break
        print('person id:', person_id)
        
        # Get list of annotations with at least one person that are not crowds
        person_anno_list = []
        for anno in tqdm.tqdm(anno_dict['annotations']):
            if (anno['category_id'] == person_id) and (anno['iscrowd'] == 0):
                _anno = copy.deepcopy(anno)
                # Remove segmentation mask
                del _anno['segmentation']
                # Add some new fields to anno
                _anno['person_id'] = _anno['id']
                _anno['is_known'] = True
                _anno['iou_thresh'] = 0.5
                # Store anno in list
                person_anno_list.append(_anno)
        print(len(person_anno_list))
        
        # Get image ids containing person annotations
        image_id_set = set()
        for person_anno in person_anno_list:
            image_id = person_anno['image_id']
            image_id_set.add(image_id)
        print(len(image_id_set))
        
        # Initialize new anno dict
        new_anno_dict = {
            'info': anno_dict['info'],
            'licenses': anno_dict['licenses'],
            'annotations': person_anno_list,
            'images': [],
        }
        
        # Get images containing persons
        for image in anno_dict['images']:
            image_id = image['id']
            if image_id in image_id_set:
                # Add cam_id field for consistency
                image['cam_id'] = -1
                new_anno_dict['images'].append(image)
                
        # Return new anno dict
        return new_anno_dict

    new_train_anno_dict = get_person_anno(train_anno_dict)
    with open(new_train_anno_path, 'w') as fp:
        json.dump(new_train_anno_dict, fp)

    new_val_anno_dict = get_person_anno(val_anno_dict)
    with open(new_val_anno_path, 'w') as fp:
        json.dump(new_val_anno_dict, fp)

    subset_list = [10, 100, 1000, 10000]
    def _get_subset_anno(image_id_list, anno_list):
        subset_anno_list = []
        for anno in tqdm.tqdm(anno_list):
            if anno['image_id'] in image_id_list:
                subset_anno_list.append(anno)
        return subset_anno_list

    def get_subset_image(anno_dict, subset_list, seed=0):
        # Shuffle images
        image_list = anno_dict['images']
        np.random.seed(seed)
        np.random.shuffle(image_list)
        
        # Get annotations
        anno_list = anno_dict['annotations']
        
        # Build anno dict template
        new_anno_dict = {
            'info': anno_dict['info'],
            'licenses': anno_dict['licenses'],
            'annotations': [],
            'images': [],
        }
        
        #
        anno_dict_dict = {}
        
        # Get subsets of images
        i = 0
        for num_subset in subset_list:
            for partition in ['train', 'val']:
                print(i)
                subset_images = image_list[i:i+num_subset]
                i += num_subset
                
                subset_image_ids = [image['id'] for image in subset_images]
                subset_annos = _get_subset_anno(subset_image_ids, anno_list)
                
                subset_anno_dict = copy.deepcopy(new_anno_dict)
                subset_anno_dict['images'] = subset_images
                subset_anno_dict['annotations'] = subset_annos
                
                partition_key = 'G{}{}'.format(num_subset, partition)
                anno_dict_dict[partition_key] = subset_anno_dict
                
        return anno_dict_dict
        
    anno_dict_dict = get_subset_image(new_train_anno_dict, subset_list)

    for partition_key, partition_anno_dict in anno_dict_dict.items():
        partition_path = f'{anno_dir}/{partition_key}.json'
        with open(partition_path, 'w') as fp:
            print(f'{partition_key}: {partition_path}')
            json.dump(partition_anno_dict, fp)

def coco2cocoa(coco_dir):
    # Anno dir
    source_anno_dir = f'{coco_dir}/annotations'
    anno_dir = f'{coco_dir}/cocoa'

    # 
    if not os.path.exists(anno_dir):
        os.makedirs(anno_dir)

    # Train anno path
    train_anno_path = f'{source_anno_dir}/instances_train2017.json'
    new_train_anno_path = f'{anno_dir}/train.json'

    # Val anno path
    val_anno_path = f'{source_anno_dir}/instances_val2017.json'
    new_val_anno_path = f'{anno_dir}/val.json'

    with open(train_anno_path, 'r') as fp:
        train_anno_dict = json.load(fp)

    with open(val_anno_path, 'r') as fp:
        val_anno_dict = json.load(fp)

    def get_person_anno(anno_dict):
        # Get list of annotations with at least one person that are not crowds
        person_anno_list = []
        for anno in tqdm.tqdm(anno_dict['annotations']):
            _anno = copy.deepcopy(anno)
            # Remove segmentation mask
            del _anno['segmentation']
            # Add some new fields to anno
            _anno['person_id'] = _anno['id']
            _anno['is_known'] = True
            _anno['iou_thresh'] = 0.5
            # Store anno in list
            person_anno_list.append(_anno)
        print(len(person_anno_list))
        
        # Get image ids containing person annotations
        image_id_set = set()
        for person_anno in person_anno_list:
            image_id = person_anno['image_id']
            image_id_set.add(image_id)
        print(len(image_id_set))
        
        # Initialize new anno dict
        new_anno_dict = {
            'info': anno_dict['info'],
            'licenses': anno_dict['licenses'],
            'annotations': person_anno_list,
            'images': [],
        }
        
        # Get images containing persons
        for image in anno_dict['images']:
            image_id = image['id']
            if image_id in image_id_set:
                # Add cam_id field for consistency
                image['cam_id'] = -1
                new_anno_dict['images'].append(image)
                
        # Return new anno dict
        return new_anno_dict

    new_train_anno_dict = get_person_anno(train_anno_dict)
    with open(new_train_anno_path, 'w') as fp:
        json.dump(new_train_anno_dict, fp)

    new_val_anno_dict = get_person_anno(val_anno_dict)
    with open(new_val_anno_path, 'w') as fp:
        json.dump(new_val_anno_dict, fp)

    subset_list = [10, 100, 1000, 10000]
    def _get_subset_anno(image_id_list, anno_list):
        subset_anno_list = []
        for anno in tqdm.tqdm(anno_list):
            if anno['image_id'] in image_id_list:
                subset_anno_list.append(anno)
        return subset_anno_list

    def get_subset_image(anno_dict, subset_list, seed=0):
        # Shuffle images
        image_list = anno_dict['images']
        np.random.seed(seed)
        np.random.shuffle(image_list)
        
        # Get annotations
        anno_list = anno_dict['annotations']
        
        # Build anno dict template
        new_anno_dict = {
            'info': anno_dict['info'],
            'licenses': anno_dict['licenses'],
            'annotations': [],
            'images': [],
        }
        
        #
        anno_dict_dict = {}
        
        # Get subsets of images
        i = 0
        for num_subset in subset_list:
            for partition in ['train', 'val']:
                print(i)
                subset_images = image_list[i:i+num_subset]
                i += num_subset
                
                subset_image_ids = [image['id'] for image in subset_images]
                subset_annos = _get_subset_anno(subset_image_ids, anno_list)
                
                subset_anno_dict = copy.deepcopy(new_anno_dict)
                subset_anno_dict['images'] = subset_images
                subset_anno_dict['annotations'] = subset_annos
                
                partition_key = 'G{}{}'.format(num_subset, partition)
                anno_dict_dict[partition_key] = subset_anno_dict
                
        return anno_dict_dict
        
    anno_dict_dict = get_subset_image(new_train_anno_dict, subset_list)

    for partition_key, partition_anno_dict in anno_dict_dict.items():
        partition_path = f'{anno_dir}/{partition_key}.json'
        with open(partition_path, 'w') as fp:
            print(f'{partition_key}: {partition_path}')
            json.dump(partition_anno_dict, fp)


# Crop person bounding boxes from cocop images
def crop_cocop():
    # Parse user args
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='/datasets/coco')
    parser.add_argument('--partition', type=str, default='persons')
    args = parser.parse_args() 

    # Set up directories
    coco_path = os.path.join(args.dataset_dir, f'coco/{args.partition}.json')
    image_dir = os.path.join(args.dataset_dir, f'{args.partition}2017')
    crop_dir = os.path.join(args.dataset_dir, f'crops/{args.partition}/person')

    if not os.path.exists(crop_dir):
        os.makedirs(crop_dir)

    # Load metadata
    with open(coco_path, 'r') as fp:
        coco_dict = json.load(fp)

    # Prep metadata
    image_dict = {}
    for image in coco_dict['images']:
        image_dict[image['id']] = image

    anno_dict = collections.defaultdict(list)
    for anno in coco_dict['annotations']:
        anno_dict[anno['image_id']].append(anno)

    # Crop all persons from a COCO image
    def _crop_image(image_path, anno_list, crop_dir):
        try:
            image = Image.open(image_path)
            for anno in anno_list:
                bbox = anno['bbox']
                x, y, w, h = bbox
                x1, y1, x2, y2 = x, y, x+w, y+h
                crop = image.crop((x1, y1, x2, y2))
                anno_id = anno['id']
                crop_file = f'{anno_id}.png'
                crop_path = os.path.join(crop_dir, crop_file)
                crop.save(crop_path)
        except:
            return 1
        else:
            return 0

    # Speed up cropping with multithreading
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for image_id in tqdm.tqdm(image_dict):
            image_file = image_dict[image_id]['file_name']
            image_path = os.path.join(image_dir, image_file)
            assert os.path.exists(image_path)
            anno_list = anno_dict[image_id]
            futures.append(executor.submit(_crop_image,
                image_path=image_path, anno_list=anno_list, crop_dir=crop_dir))
        for future in tqdm.tqdm(concurrent.futures.as_completed(futures)):
            pass


# Main function
def main():
    # Function imports
    import argparse

    # Parse user args
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='/datasets/coco')
    parser.add_argument('--partition', type=str, default='persons')
    args = parser.parse_args() 

    # Build the COCO dataset for the COCOPersons subset
    if args.partition == 'persons':
        coco2cocop(args.dataset_dir)
    elif args.partition == 'all':
        coco2cocoa(args.dataset_dir)
    else: raise NotImplementedError(args.partition)

    # Build the sample ablation partitions for COCOPersons (<100)
    cocop_sample_ablation(args.dataset_dir)

    # Build the sample ablation partitions for COCOPersons (>100)
    cocop_add_ablation(args.dataset_dir)


# Module call
if __name__ == '__main__':
    main()
