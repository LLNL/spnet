# Global imports
import os
import yaml
import random
import numpy as np
import pandas as pd
## torch
import torch
import torch.utils.data

# Package imports
## SeqNeXt model
from osr.models.seqnext import SeqNeXt
## engine
from osr.engine import transform
from osr.engine.group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
## data
from osr.data import det_utils
from osr.engine.group_by_pid import GroupedPIDBatchSamplerEdgeCover, create_pid_groups


# Helper function to collate data
def collate_fn(batch):
    return tuple(zip(*batch))


# Helper function to move data to GPU
def to_device(images, targets, device):
    images = [image.to(device) for image in images]
    for t in targets:
        t["boxes"] = t["boxes"].to(device)
        t["labels"] = t["labels"].to(device)
    return images, targets


# Helper to delete keys from torch module state dict
def _del_key(state_dict, key):
    if key in state_dict:
        del state_dict[key]


# YAML config loader function
def load_config(path, tuple_key_list=None):
    # Load config dict from YAML
    with open(path, 'r') as fp:
        try:
            config = yaml.safe_load(fp)
        except yaml.YAMLError as exc:
            print(exc)
            raise

    # Params that have tuple type
    if tuple_key_list is None:
        tuple_key_list = config['tuple_key_list']
        del config['tuple_key_list']
    elif 'tuple_key_list' in config:
        del config['tuple_key_list']
    
    # Convert lists to tune.grid_search
    proc_config = {}
    for key, val in config.items():
        if key in tuple_key_list:
            if type(val) == list:
                raise Exception
            else:
                proc_config[key] = eval(val)
        else:
            # Exception to handle loading of train dataset dictionaries
            if type(val) == list:
                raise Exception
            else:
                proc_config[key] = val

    # Return processed config dict
    return proc_config, tuple_key_list


# Function for reproducible seeding of DataLoader
def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_test_loader(config):
    # Use ImageNet stats to standardize the data
    stat_dict = {
        'mean': config['image_mean'],
        'std': config['image_std'],
    }

    # Set transform
    ## IFN transform
    if config['aug_mode'] == 'wrs':
        test_transform = transform.get_transform_wrs(train=False, stat_dict=stat_dict)
    elif config['aug_mode'] == 'wrsrrc':
        test_transform = transform.get_transform_wrsrrc(train=False, stat_dict=stat_dict)
    elif config['aug_mode'] == 'rwrs':
        test_transform = transform.get_transform_rwrs(train=False, stat_dict=stat_dict)
    elif config['aug_mode'] == 'rwrsrsc':
        test_transform = transform.get_transform_rwrsrsc(train=False, stat_dict=stat_dict)
    elif config['aug_mode'] == 'lursc':
        test_transform = transform.get_transform_lursc(train=False, stat_dict=stat_dict)
    elif config['aug_mode'] == 'lurrc':
        test_transform = transform.get_transform_lurrc(train=False, stat_dict=stat_dict)
    elif config['aug_mode'] == 'rrc':
        test_transform = transform.get_transform_rrc(train=False, stat_dict=stat_dict)
    elif config['aug_mode'] == 'rrc_scale':
        test_transform = transform.get_transform_rrc_scale(train=False, stat_dict=stat_dict)
    elif config['aug_mode'] == 'rrcj':
        test_transform = transform.get_transform_rrcj(train=False, stat_dict=stat_dict)
    elif config['aug_mode'] == 'rrc2':
        test_transform = transform.get_transform_rrc2(train=False, stat_dict=stat_dict)
    elif config['aug_mode'] == 'lmspad':
        test_transform = transform.get_transform_lmspad(train=False, stat_dict=stat_dict)
    elif config['aug_mode'] == 'rsc':
        test_transform = transform.get_transform_rsc(train=False, stat_dict=stat_dict)
    elif config['aug_mode'] == 'rs':
        test_transform = transform.get_transform_rs(train=False, stat_dict=stat_dict)
    elif config['aug_mode'] == 'in1k':
        test_transform = transform.get_transform_in1k(train=False, stat_dict=stat_dict)
    elif config['aug_mode'] == 'in1k2':
        test_transform = transform.get_transform_in1k2(train=False, stat_dict=stat_dict)
    elif config['aug_mode'] == 'in1k3':
        test_transform = transform.get_transform_in1k3(train=False, stat_dict=stat_dict)
    elif config['aug_mode'] == 'rrcin1k':
        test_transform = transform.get_transform_rrcin1k(train=False, stat_dict=stat_dict)

    assert len(config['test_dataset']) == 1, 'Only one test dataset permitted.'

    test_dataset_name = list(config['test_dataset'].keys())[0]
    test_dataset_dict = config['test_dataset'][test_dataset_name]

    # Get pid lookup
    pid_lookup_dict, _ = det_utils.get_coco_labels(
        test_dataset_dict['dir'],
        dataset_name=test_dataset_name,
        image_set=test_dataset_dict['subset'],
    )

    # Get dataset
    test_dataset = det_utils.get_coco_dataset(
        test_dataset_dict['dir'],
        dataset_name=test_dataset_name,
        image_set=test_dataset_dict['subset'],
        transforms=test_transform,
        pid_lookup_dict=pid_lookup_dict,
        ssl=False)

    # Get sampler
    retrieval_dir = os.path.join(test_dataset_dict['dir'], 'retrieval')
    test_sampler = det_utils.TestSampler(test_dataset_dict['subset'],
        test_dataset, retrieval_dir,
        config['retrieval_name_list'])

    # Get loader
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config['val_batch_size'],
        sampler=test_sampler, num_workers=config['workers'], persistent_workers=True,
        collate_fn=collate_fn)

    # Return loader
    return test_loader


def get_train_loader(config, rank=0, world_size=1, partition='train'):
    # Use ImageNet stats to standardize the data
    stat_dict = {
        'mean': config['image_mean'],
        'std': config['image_std'],
    }

    # Set transform
    ## IFN transform
    if config['aug_mode'] == 'wrs':
        train_transform = transform.get_transform_wrs(train=True, stat_dict=stat_dict)
    elif config['aug_mode'] == 'wrsrrc':
        train_transform = transform.get_transform_wrsrrc(train=True, stat_dict=stat_dict,
            rfc_prob=config['aug_rfc_prob'], rsc_prob=config['aug_rsc_prob'],
            rbsc_er=config['aug_rbsc_er'], crop_res=config['aug_crop_res'])
    elif config['aug_mode'] == 'rwrs':
        train_transform = transform.get_transform_rwrs(train=True, stat_dict=stat_dict)
    elif config['aug_mode'] == 'rwrsrsc':
        train_transform = transform.get_transform_rwrsrsc(train=True, stat_dict=stat_dict)
    elif config['aug_mode'] == 'lursc':
        train_transform = transform.get_transform_lursc(train=True, stat_dict=stat_dict)
    elif config['aug_mode'] == 'lurrc':
        train_transform = transform.get_transform_lurrc(train=True, stat_dict=stat_dict)
    elif config['aug_mode'] == 'rrc':
        train_transform = transform.get_transform_rrc(train=True, stat_dict=stat_dict,
            rfc_prob=config['aug_rfc_prob'], rsc_prob=config['aug_rsc_prob'],
            rbsc_er=config['aug_rbsc_er'], crop_res=config['aug_crop_res'])
    elif config['aug_mode'] == 'rrc_scale':
        train_transform = transform.get_transform_rrc_scale(train=True, stat_dict=stat_dict,
            rfc_prob=config['aug_rfc_prob'], rsc_prob=config['aug_rsc_prob'],
            rbsc_er=config['aug_rbsc_er'], crop_res=config['aug_crop_res'])
    elif config['aug_mode'] == 'rrcj':
        train_transform = transform.get_transform_rrcj(train=True, stat_dict=stat_dict,
            rfc_prob=config['aug_rfc_prob'], rsc_prob=config['aug_rsc_prob'],
            rbsc_er=config['aug_rbsc_er'], crop_res=config['aug_crop_res'])
    elif config['aug_mode'] == 'rrc2':
        train_transform = transform.get_transform_rrc2(train=True, stat_dict=stat_dict,
            rfc_prob=config['aug_rfc_prob'], rsc_prob=config['aug_rsc_prob'],
            rbsc_er=config['aug_rbsc_er'], crop_res=config['aug_crop_res'])
    elif config['aug_mode'] == 'lmspad':
        train_transform = transform.get_transform_rrc2(train=True, stat_dict=stat_dict,
            rfc_prob=config['aug_rfc_prob'], rsc_prob=config['aug_rsc_prob'],
            rbsc_er=config['aug_rbsc_er'], crop_res=config['aug_crop_res'])
    elif config['aug_mode'] == 'rsc':
        train_transform = transform.get_transform_rsc(train=True, stat_dict=stat_dict,
            rfc_prob=config['aug_rfc_prob'], rsc_prob=config['aug_rsc_prob'],
            rbsc_er=config['aug_rbsc_er'], crop_res=config['aug_crop_res'])
    elif config['aug_mode'] == 'rs':
        train_transform = transform.get_transform_rs(train=True, stat_dict=stat_dict)
    elif config['aug_mode'] == 'in1k':
        train_transform = transform.get_transform_in1k(train=True, stat_dict=stat_dict)
    elif config['aug_mode'] == 'in1k2':
        train_transform = transform.get_transform_in1k2(train=True, stat_dict=stat_dict)
    elif config['aug_mode'] == 'in1k3':
        train_transform = transform.get_transform_in1k3(train=True, stat_dict=stat_dict)
    elif config['aug_mode'] == 'rrcin1k':
        train_transform = transform.get_transform_rrcin1k(train=True, stat_dict=stat_dict,
            rfc_prob=config['aug_rfc_prob'], rsc_prob=config['aug_rsc_prob'],
            rbsc_er=config['aug_rbsc_er'], crop_res=config['aug_crop_res'])

    # Get dataset
    ## Build SSL anno params dict
    if config['use_ssl']:
        ssl_anno_params = {
            'anno_method': config['ssl_anno_method'],
            'num_anno': config['ssl_num_anno'],
            'min_width': config['ssl_min_width'],
            'max_width': config['ssl_max_width'],
            'min_ar': config['ssl_min_ar'],
            'max_ar': config['ssl_max_ar'],
            'iou_thresh': config['ssl_iou_thresh'],
            'filter_monotone': config['ssl_filter_monotone'],
            'filter_duplicate': config['ssl_filter_duplicate'],
        }
    else:
        ssl_anno_params = {} 
    ## Load train dataset even in test only mode, because we need to get the pid_lookup_dict to load the correct OIM
    partition_dataset = f'{partition}_dataset'
    pid_lookup_dict_dict = {}
    aspect_ratios_dict_dict = {}
    for dataset_name, dataset_dict in config[partition_dataset].items():
        pid_lookup_dict, aspect_ratios_dict = det_utils.get_coco_labels(
            dataset_dict['dir'],
            dataset_name=dataset_name,
            image_set=dataset_dict['subset'],
        )
        pid_lookup_dict_dict[dataset_name] = pid_lookup_dict
        aspect_ratios_dict_dict[dataset_name] = aspect_ratios_dict

    ## Show aggregated metadata for all datasets
    info_dict = {'total': {
        '# Images': 0,
        '# Known ID': 0,
        '# Unknown ID': 0,
    }}
    for dataset_name, pid_lookup_dict in pid_lookup_dict_dict.items():
        # Show partition information
        info_dict.update({dataset_name: {
            '# Images': pid_lookup_dict['num_img'],
            '# Known ID': pid_lookup_dict['num_pid'],
            '# Unknown ID': pid_lookup_dict['num_uid'],
        }})
        info_dict['total']['# Images'] += pid_lookup_dict['num_img']
        info_dict['total']['# Known ID'] += pid_lookup_dict['num_pid']
        info_dict['total']['# Unknown ID'] += pid_lookup_dict['num_uid']
    info_df = pd.DataFrame(info_dict).T
    print(info_df)

    ## Merge pid lookups
    tot_pid, tot_uid, tot_img = 0, 0, 0
    for dataset_name, pid_lookup_dict in pid_lookup_dict_dict.items():
        ### Update all pids
        for pid in pid_lookup_dict['pid_lookup']:
            pid_lookup_dict['pid_lookup'][pid] += tot_pid + tot_uid
        ### Update known pids
        for pid in pid_lookup_dict['label_lookup']:
            pid_lookup_dict['label_lookup'][pid] += tot_pid
        ### Update total counts
        tot_pid += pid_lookup_dict['num_pid']
        tot_uid += pid_lookup_dict['num_uid'] 
        tot_img += pid_lookup_dict['num_img']
    ## Update total counts
    for dataset_name, pid_lookup_dict in pid_lookup_dict_dict.items():
        pid_lookup_dict['num_pid'] = tot_pid
        pid_lookup_dict['num_uid'] = tot_uid
        pid_lookup_dict['num_img'] = tot_img
    ## Check counts
    assert tot_pid == info_dict['total']['# Known ID']
    assert tot_uid == info_dict['total']['# Unknown ID']
    assert tot_img == info_dict['total']['# Images']

    ## For SSL: set pid lookups to None
    if config['use_ssl']:
        for k in pid_lookup_dict_dict:
            pid_lookup_dict_dict[k] = None

    ## Build dataset objects
    dataset_list = []
    for dataset_name, dataset_dict in config[partition_dataset].items():
        dataset = det_utils.get_coco_dataset(
            dataset_dict['dir'],
            dataset_name=dataset_name,
            image_set=dataset_dict['subset'],
            transforms=train_transform,
            pid_lookup_dict=pid_lookup_dict_dict[dataset_name],
            ssl=config['use_ssl'], ssl_anno_params=ssl_anno_params)
        dataset_list.append(dataset)

    ## Build train dataset for list of datasets
    if len(dataset_list) > 1:
        train_dataset = torch.utils.data.ConcatDataset(dataset_list)
    else:
        train_dataset = dataset_list[0]
    
    # Train sampler
    print("Creating data loaders")
    if config['debug'] or config['use_ssl']:
        print('WARNING: sequential sampler for debugging')
        train_sampler = torch.utils.data.SequentialSampler(train_dataset)
    else:
        print('Using random sampler...')
        ## Control randomness
        if config['use_random_seed']:
            g_rs = torch.Generator()
            g_rs.manual_seed(config['random_seed'])
        else:
            g_rs = None
        ## Build train sampler
        train_sampler = torch.utils.data.RandomSampler(train_dataset, generator=g_rs)

    # Determine aspect ratio grouping based on dataset
    if 'cuhk' in config[f'{partition}_dataset']:
        if config['aspect_ratio_grouping']:
            print('Using aspect ratio batch sampler...')
            ## Group into two bins: wide and tall
            ### Saves considerable memory with WindowResize transform, allowing for larger batch size
            aspect_ratio_group_factor = 0
        else:
            aspect_ratio_group_factor = -1
    elif config[f'{partition}_dataset'] == 'prw':
        ## AR grouping does not benefit PRW: 5/6 cameras have same image size
        aspect_ratio_group_factor = -1
    else:
        aspect_ratio_group_factor = -1

    # Setup aspect ratio batch sampler
    if (config['use_ssl']) or (config['sampler_mode'] == 'repeat'):
        num_views = config['sampler_num_repeat']
        assert config['batch_size']%num_views == 0
        train_batch_sampler = det_utils.SSLBatchSampler(train_sampler,
            config['batch_size']//num_views, num_views,
            rank=rank, num_replicas=world_size)
    elif config['sampler_mode'] == 'pair':
        print('==> Using GroupedPIDBatchSamplerEdgeCover')
        if config['aspect_ratio_grouping']:
            print('+ Grouping aspect ratios')
            raise NotImplementedError('Not implemented for multi datasets yet...')
        else:
            print('+ NOT Grouping aspect ratios')
            aspect_ratios_dict = None
        image_ids, person_ids, is_known = create_pid_groups(train_dataset)
        train_batch_sampler = GroupedPIDBatchSamplerEdgeCover(train_sampler,
            person_ids, image_ids, is_known,
            aspect_ratios_dict=aspect_ratios_dict,
            num_pid=config['batch_size']//2, img_per_pid=2,
            num_replicas=1, rank=0, shuffle=True, seed=0, lookup_path=None)
    else:
        # Setup aspect ratio batch sampler
        if aspect_ratio_group_factor >= 0:
            print('==> Aspect ratio grouping!')
            group_ids = create_aspect_ratio_groups(
                train_dataset, k=aspect_ratio_group_factor)
            train_batch_sampler = GroupedBatchSampler(
                train_sampler, group_ids, config['batch_size'])
        else:
            train_batch_sampler = torch.utils.data.BatchSampler(
                train_sampler, config['batch_size'], drop_last=True)

    # Set up train loader
    ## Control randomness
    if config['use_random_seed']:
        g_dl = torch.Generator()
        g_dl.manual_seed(config['random_seed'])
        worker_init_fn = _seed_worker
    else:
        g_dl = None
        worker_init_fn = None
    ## Build train loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_sampler=train_batch_sampler,
        num_workers=config['workers'], persistent_workers=False,
        collate_fn=collate_fn, prefetch_factor=2,
        generator=g_dl, worker_init_fn=worker_init_fn)

    #
    return train_loader, tot_pid


