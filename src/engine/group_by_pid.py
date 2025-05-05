import bisect
import os
import pickle
from collections import defaultdict
import collections
import copy
from itertools import repeat, chain
import math
import numpy as np
from pprint import pprint

import torch
import torch.utils.data
from torch.utils.data.sampler import BatchSampler, Sampler
from torch.utils.model_zoo import tqdm
import torchvision

from PIL import Image

# Edge cover packages
import networkx as nx
import random


def _repeat_to_at_least(iterable, n):
    repeat_times = math.ceil(n / len(iterable))
    repeated = chain.from_iterable(repeat(iterable, repeat_times))
    return list(repeated)


class GroupedPIDBatchSampler(BatchSampler):
    """
    Wraps another sampler to yield a mini-batch of indices.
    It enforces that the batch only contain elements from the same group.
    It also tries to provide mini-batches which follows an ordering which is
    as close as possible to the ordering from the original sampler.
    Arguments:
        sampler (Sampler): Base sampler.
        group_ids (list[int]): If the sampler produces indices in range [0, N),
            `group_ids` must be a list of `N` ints which contains the group id of each sample.
            The group ids must be a continuous set of integers starting from
            0, i.e. they must be in the range [0, num_groups).
        batch_size (int): Size of mini-batch.
    """
    def __init__(self, sampler, person_ids, num_pid, img_per_pid, max_single=4,
        num_replicas=None, rank=None, shuffle=True, seed=0):
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

        # Params
        self.sampler = sampler
        self.person_ids = person_ids
        self.num_pid = num_pid
        self.img_per_pid = img_per_pid
        self.batch_size = self.num_pid * self.img_per_pid
        self.max_single = max_single

        # XXX: Take subset of sampler for speedier run
        #self.sampler = list(self.sampler)[:100]

        ## Compute the master img group set dict
        ## Make groups of pids
        # Compute image, pid lookup dicts
        img_pid_dict = collections.defaultdict(set)
        for img in self.sampler:
            pid_list = self.person_ids[img]
            for pid in pid_list:
                if pid.startswith('p'):
                    img_pid_dict[img].add(pid)

        # Lookup dict for image sets
        img_set_lookup_dict = collections.defaultdict(set)
        for img1, pid_set1 in img_pid_dict.items():
            for img2 in img_pid_dict:
                if img2 > img1:
                    pid_set2 = img_pid_dict[img2]
                    if pid_set1.intersection(pid_set2):
                        img_set_lookup_dict[img1].add(img2)
                        img_set_lookup_dict[img2].add(img1)
        for img in img_pid_dict:
            if img not in img_set_lookup_dict:
                img_set_lookup_dict[img] = set()

        count_dict = collections.defaultdict(int)
        for img, img_set in sorted(img_set_lookup_dict.items(), key=lambda x: len(x[1]), reverse=True):
            count_dict[len(img_set)] += 1
        pprint(count_dict)
        #exit()
        self.img_set_lookup_dict = img_set_lookup_dict
        self.img_pid_dict = img_pid_dict

    def __iter__(self):
        return iter(self.replica_list)

    def __len__(self):
        return (len(self.sampler) // self.batch_size) // self.num_replicas

    def set_epoch(self, epoch):
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Arguments:
            epoch (int): Epoch number.
        """
        ### Set the epoch
        self.epoch = epoch

        ### Prep the batch and determine its length
        # Set random seed
        np.random.seed(self.seed + self.epoch)

        # Shuffle the index list
        sampler_idx_list = list(self.sampler)
        np.random.shuffle(sampler_idx_list)

        # Keep track of stuff
        unique_pid_set = set()
        num_single_groups, num_tot_groups = 0, 0
        num_dup_batch, num_tot_batch = 0, 0
        num_bad_batch = 0
        num_batches = 0
        num_buffer_single = 0
        tot_img = len(self.sampler)
        group_count_dict = collections.defaultdict(int)

        # Build the batch list
        buffer_list = []
        batch_list = []
        leftover_set = set()
        for idx1 in sampler_idx_list:
            if idx1 in unique_pid_set:
                continue
            img_list = list(self.img_set_lookup_dict[idx1])
            if len(img_list) > 0:
                idx2 = np.random.choice(img_list)
                if len(buffer_list) == (self.batch_size - 1):
                    leftover_set.add(buffer_list.pop())
                buffer_list.extend([idx1, idx2])
            else:
                if num_buffer_single >= self.max_single:
                    leftover_set.add(idx1)
                else:
                    buffer_list.extend([idx1])
                    num_single_groups += 1
                    num_buffer_single += 1
            num_tot_groups += 1

            # Check if any pid appears twice
            if len(buffer_list) == self.batch_size:
                buffer_pid_list = []
                for elem in buffer_list:
                    buffer_pid_list.extend(list(self.img_pid_dict[elem]))
                # Check if all pids are unique: bad
                if len(buffer_pid_list) == len(set(buffer_pid_list)):
                    leftover_set.add(buffer_list.pop())
                    buffer_list.append(buffer_list[-1])
                    num_bad_batch += 1
                    raise Exception
                # Check unique image ids
                else:
                    buffer_img_set = set(buffer_list)
                    unique_pid_set.update(buffer_img_set)
                    if len(buffer_img_set) < len(buffer_list):
                        num_buffer_dup = self.batch_size - len(buffer_img_set)
                        if len(leftover_set) >= num_buffer_dup:
                            buffer_list = list(set(buffer_list))
                            print('==> Num buffer dup: {}'.format(num_buffer_dup))
                            for _ in range(num_buffer_dup):
                                buffer_list.append(leftover_set.pop())
                                num_single_groups += 1
                            assert len(buffer_list) == self.batch_size
                        else:
                            num_dup_batch += 1
                num_tot_batch += 1

                # Count final selected buffer pids
                final_buffer_pid_list = []
                for elem in buffer_list:
                    final_buffer_pid_list.extend(list(self.img_pid_dict[elem]))

                # Count how many groups of each size there are
                unique_pid_arr, unique_count_arr = np.unique(final_buffer_pid_list, return_counts=True)
                for count in unique_count_arr:
                    group_count_dict[count] += 1

                #print(buffer_list)
                #yield buffer_list
                batch_list.append(buffer_list.copy())

                buffer_list = []
                num_buffer_single = 0

            if num_tot_batch == (len(self) * self.num_replicas):
                break

        print('Num leftover IDs: {}'.format(len(leftover_set)))
        print('Num unique ImageID used: {}/{}'.format(len(unique_pid_set), tot_img))
        print('Num singleton groups: {}/{}'.format(num_single_groups, num_tot_groups))
        print('Num duplicate batches: {}/{}'.format(num_dup_batch, num_tot_batch))
        print('Num bad batches: {}/{}'.format(num_bad_batch, num_tot_batch))
        print('Group counts:')
        for count, num in sorted(group_count_dict.items(), key=lambda x: x[0]):
            print('{}: {}'.format(count, num))
        # now we have run out of elements that satisfy
        # the group criteria, let's return the remaining
        # elements so that the size of the sampler is
        # deterministic
        expected_num_batches = len(self) * self.num_replicas
        num_remaining = expected_num_batches - num_tot_batch
        #print('Num batches remaining: {}'.format(num_remaining))
        #assert num_remaining == 0
        # Get the batches to be used for this process
        print('==> SET THE REPLICA LIST')
        self.replica_list = batch_list[self.rank::self.num_replicas]

class GroupedPIDBatchSampler2(BatchSampler):
    """
    Wraps another sampler to yield a mini-batch of indices.
    It enforces that the batch only contain elements from the same group.
    It also tries to provide mini-batches which follows an ordering which is
    as close as possible to the ordering from the original sampler.
    Arguments:
        sampler (Sampler): Base sampler.
        group_ids (list[int]): If the sampler produces indices in range [0, N),
            `group_ids` must be a list of `N` ints which contains the group id of each sample.
            The group ids must be a continuous set of integers starting from
            0, i.e. they must be in the range [0, num_groups).
        batch_size (int): Size of mini-batch.
    """
    def __init__(self, sampler, person_ids, num_pid, img_per_pid, max_single=4,
        num_replicas=None, rank=None, shuffle=True, seed=0):
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

        # Params
        self.sampler = sampler
        self.person_ids = person_ids
        self.num_pid = num_pid
        self.img_per_pid = img_per_pid
        self.batch_size = self.num_pid * self.img_per_pid
        self.max_single = max_single

        # Don't use all of sampler
        #self.sampler = list(self.sampler)[:1000]

        ## Compute the master img group set dict
        ## Make groups of pids
        # Compute image, pid lookup dicts
        img_pid_dict = collections.defaultdict(set)
        for img in self.sampler:
            pid_list = self.person_ids[img]
            for pid in pid_list:
                if pid.startswith('p'):
                    img_pid_dict[img].add(pid)

        # Lookup dict for image sets
        img_set_lookup_dict = collections.defaultdict(set)
        for img1, pid_set1 in img_pid_dict.items():
            for img2 in img_pid_dict:
                if img2 > img1:
                    pid_set2 = img_pid_dict[img2]
                    if pid_set1.intersection(pid_set2):
                        img_set_lookup_dict[img1].add(img2)
                        img_set_lookup_dict[img2].add(img1)
        for img in img_pid_dict:
            if img not in img_set_lookup_dict:
                img_set_lookup_dict[img] = set()

        count_dict = collections.defaultdict(int)
        for img, img_set in sorted(img_set_lookup_dict.items(), key=lambda x: len(x[1]), reverse=True):
            count_dict[len(img_set)] += 1
        pprint(count_dict)
        #exit()
        self.img_set_lookup_dict = img_set_lookup_dict
        self.img_pid_dict = img_pid_dict

    def __iter__(self):
        return iter(self.replica_list)

    def __len__(self):
        return (len(self.sampler) // self.batch_size) // self.num_replicas

    def set_epoch(self, epoch):
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Arguments:
            epoch (int): Epoch number.
        """
        ### Set the epoch
        self.epoch = epoch

        ### Prep the batch and determine its length
        # Set random seed
        np.random.seed(self.seed + self.epoch)

        # Shuffle the index list
        sampler_idx_list = list(self.sampler)
        np.random.shuffle(sampler_idx_list)

        # Keep track of stuff
        unique_pid_set = set()
        num_single_groups, num_tot_groups = 0, 0
        num_dup_batch, num_tot_batch = 0, 0
        num_bad_batch = 0
        num_batches = 0
        num_buffer_single = 0
        tot_img = len(self.sampler)
        group_count_dict = collections.defaultdict(int)

        # Build the batch list
        buffer_list = []
        batch_list = []
        leftover_set = set()
        for idx1 in sampler_idx_list:
            # XXX: this line allowed more unique coverage
            #if idx1 in unique_pid_set:
            #    continue
            img_list = list(self.img_set_lookup_dict[idx1])
            # Num available to sample
            num_avail = len(img_list)
            # XXX: this line means skipping small groups 
            if num_avail < self.img_per_pid:
                continue
            # How many items are left in the batch to fill?
            num_remain = self.batch_size - len(buffer_list)
            # There should be more than 0 remaining
            assert num_remain > 0
            if num_avail > 0:
                # Sample img_per_pid by default, otherwise min of num remaining and num available
                num_sample = min(self.img_per_pid, num_avail, num_remain) - 1
                # Sample from possibilities for this idx
                idx2_list = np.random.choice(img_list, num_sample, replace=False).tolist()
                # Use the original idx
                idx2_list.append(idx1)
                buffer_list.extend(idx2_list)
            else:
                if num_buffer_single >= self.max_single:
                    leftover_set.add(idx1)
                else:
                    buffer_list.extend([idx1])
                    num_single_groups += 1
                    num_buffer_single += 1
            num_tot_groups += 1

            # Check if any pid appears twice
            if len(buffer_list) == self.batch_size:
                buffer_pid_list = []
                for elem in buffer_list:
                    buffer_pid_list.extend(list(self.img_pid_dict[elem]))
                # Check if all pids are unique: bad
                if len(buffer_pid_list) == len(set(buffer_pid_list)):
                    leftover_set.add(buffer_list.pop())
                    buffer_list.append(buffer_list[-1])
                    num_bad_batch += 1
                    #raise Exception
                # Check unique image ids
                else:
                    buffer_img_set = set(buffer_list)
                    if len(buffer_img_set) < len(buffer_list):
                        num_buffer_dup = self.batch_size - len(buffer_img_set)
                        if len(leftover_set) >= num_buffer_dup:
                            buffer_list = list(set(buffer_list))
                            print('==> Num buffer dup: {}'.format(num_buffer_dup))
                            for _ in range(num_buffer_dup):
                                buffer_list.append(leftover_set.pop())
                                num_single_groups += 1
                            assert len(buffer_list) == self.batch_size
                        else:
                            num_dup_batch += 1
                num_tot_batch += 1

                # Count final selected images
                buffer_img_set = set(buffer_list)
                unique_pid_set.update(buffer_img_set)

                # Count final selected buffer pids
                final_buffer_pid_list = []
                for elem in buffer_list:
                    final_buffer_pid_list.extend(list(self.img_pid_dict[elem]))

                # Count how many groups of each size there are
                unique_pid_arr, unique_count_arr = np.unique(final_buffer_pid_list, return_counts=True)
                for count in unique_count_arr:
                    group_count_dict[count] += 1

                #print(buffer_list)
                #yield buffer_list
                batch_list.append(buffer_list.copy())

                buffer_list = []
                num_buffer_single = 0

            if num_tot_batch == (len(self) * self.num_replicas):
                break

        print('Num leftover IDs: {}'.format(len(leftover_set)))
        print('Num unique ImageID used: {}/{}'.format(len(unique_pid_set), tot_img))
        print('Num singleton groups: {}/{}'.format(num_single_groups, num_tot_groups))
        print('Num duplicate batches: {}/{}'.format(num_dup_batch, num_tot_batch))
        print('Num bad batches: {}/{}'.format(num_bad_batch, num_tot_batch))
        print('Group counts:')
        for count, num in sorted(group_count_dict.items(), key=lambda x: x[0]):
            print('{}: {}'.format(count, num))
        # now we have run out of elements that satisfy
        # the group criteria, let's return the remaining
        # elements so that the size of the sampler is
        # deterministic
        expected_num_batches = len(self) * self.num_replicas
        num_remaining = expected_num_batches - num_tot_batch
        #print('Num batches remaining: {}'.format(num_remaining))
        #assert num_remaining == 0
        # Get the batches to be used for this process
        print('==> SET THE REPLICA LIST')
        self.replica_list = batch_list[self.rank::self.num_replicas]
        # Return the unique pid set
        return unique_pid_set

class GroupedPIDBatchSampler3(BatchSampler):
    """
    Wraps another sampler to yield a mini-batch of indices.
    It enforces that the batch only contain elements from the same group.
    It also tries to provide mini-batches which follows an ordering which is
    as close as possible to the ordering from the original sampler.
    Arguments:
        sampler (Sampler): Base sampler.
        group_ids (list[int]): If the sampler produces indices in range [0, N),
            `group_ids` must be a list of `N` ints which contains the group id of each sample.
            The group ids must be a continuous set of integers starting from
            0, i.e. they must be in the range [0, num_groups).
        batch_size (int): Size of mini-batch.
    """
    def __init__(self, sampler, person_ids, image_ids, num_pid, img_per_pid, max_single=4,
        num_replicas=None, rank=None, shuffle=True, seed=0, lookup_path=None, split_batch=False):
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

        # Params
        self.sampler = sampler
        self.person_ids = person_ids
        self.image_ids = image_ids
        self.num_pid = num_pid
        self.img_per_pid = img_per_pid
        self.batch_size = self.num_pid * self.img_per_pid
        self.max_single = max_single
        self.split_batch = split_batch

        # Don't use all of sampler
        #self.sampler = list(self.sampler)[:1000]

        if os.path.exists(lookup_path):
            print('==> Reading lookup file...')
            with open(lookup_path, 'rb') as fp:
                lookup_dict = pickle.load(fp)
                self.img_unk_dict = lookup_dict['img_unk_dict']
                self.pid_set = lookup_dict['pid_set']
                self.img_set_lookup_dict = lookup_dict['img_set_lookup_dict']
                self.img_pid_dict = lookup_dict['img_pid_dict']        
        else:
            print('==> Writing lookup file...')
            ## Compute the master img group set dict
            ## Make groups of pids
            # Compute image, pid lookup dicts
            img_pid_dict = collections.defaultdict(set)
            img_unk_dict = collections.defaultdict(set)
            print('sampler:', self.sampler)
            for img in self.sampler:
                print('img:', img)
                pid_list = self.person_ids[img]
                for pid in pid_list:
                    if pid.startswith('p'):
                        img_pid_dict[img].add(pid)
                    elif pid.startswith('u'):
                        img_unk_dict[img].add(pid)
                    else: raise Exception
            self.img_unk_dict = img_unk_dict

            # Lookup dict for image sets
            self.pid_set = set()
            img_set_lookup_dict = collections.defaultdict(set)
            for img1, pid_set1 in img_pid_dict.items():
                # Count unique img/pid combos
                for pid1 in pid_set1:
                    self.pid_set.add((img1, pid1))    
                #
                for img2 in img_pid_dict:
                    if img2 > img1:
                        pid_set2 = img_pid_dict[img2]
                        if pid_set1.intersection(pid_set2):
                            img_set_lookup_dict[img1].add(img2)
                            img_set_lookup_dict[img2].add(img1)
            for img in img_pid_dict:
                if img not in img_set_lookup_dict:
                    img_set_lookup_dict[img] = set()

            count_dict = collections.defaultdict(int)
            for img, img_set in sorted(img_set_lookup_dict.items(), key=lambda x: len(x[1]), reverse=True):
                count_dict[len(img_set)] += 1
            pprint(count_dict)

            self.img_set_lookup_dict = img_set_lookup_dict
            self.img_pid_dict = img_pid_dict

            # Build lookup dict
            lookup_dict = {
                'img_unk_dict': self.img_unk_dict,
                'pid_set': self.pid_set,
                'img_set_lookup_dict': self.img_set_lookup_dict,
                'img_pid_dict': self.img_pid_dict,
            }

            # Save lookup dict to disk
            with open(lookup_path, 'wb') as fp:
                pickle.dump(lookup_dict, fp)

    def __iter__(self):
        return iter(self.replica_list)

    def __len__(self):
        if self.split_batch:
            return 2 * (len(self.sampler) // self.batch_size) // self.num_replicas
        else:
            return (len(self.sampler) // self.batch_size) // self.num_replicas

    def set_epoch(self, epoch):
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Arguments:
            epoch (int): Epoch number.
        """
        ### Set the epoch
        self.epoch = epoch

        ### Prep the batch and determine its length
        # Set random seed
        np.random.seed(self.seed + self.epoch)

        # Shuffle the index list
        sampler_idx_list = list(self.sampler)
        np.random.shuffle(sampler_idx_list)

        # Keep track of stuff
        unique_pid_set = set()
        unique_img_set = set()
        final_unique_img_set = set()
        num_single_groups, num_tot_groups = 0, 0
        num_dup_batch, num_tot_batch = 0, 0
        num_bad_batch = 0
        num_batches = 0
        num_buffer_single = 0
        tot_img = len(self.sampler)
        group_count_dict = collections.defaultdict(int)
        img_count_dict = collections.defaultdict(int)

        # Build the batch list
        buffer_list = []
        batch_list = []
        leftover_set = set()
        for idx1 in sampler_idx_list:
            # XXX: this line allowed more unique coverage
            if idx1 in unique_img_set:
                continue
            img_list = list(self.img_set_lookup_dict[idx1])
            # Num available to sample
            num_avail = len(img_list)
            # XXX: this line means skipping small groups 
            #if (num_avail + 1) < self.img_per_pid:
            #    continue
            # How many items are left in the batch to fill?
            num_remain = self.batch_size - len(buffer_list)
            # XXX: this line means skipping small groups 
            #if ((num_avail + 1) < self.img_per_pid) and ((num_avail + 1) != num_remain):
            #    continue
            # There should be more than 0 remaining
            assert num_remain > 0
            if num_avail > 0:
                # Sample img_per_pid by default, otherwise min of num remaining and num available
                num_sample = min(self.img_per_pid, num_avail+1, num_remain) - 1
                #print(self.img_per_pid, num_avail+1, num_remain)
                # Sample from possibilities for this idx
                idx2_list = np.random.choice(img_list, num_sample, replace=False).tolist()
                # Use the original idx
                idx2_list.append(idx1)
                buffer_list.extend(idx2_list)
            else:
                if num_buffer_single >= self.max_single:
                    leftover_set.add(idx1)
                else:
                    buffer_list.extend([idx1])
                    num_single_groups += 1
                    num_buffer_single += 1
            num_tot_groups += 1

            # Check if batch is ready, and make final changes to it
            if len(buffer_list) == self.batch_size:
                buffer_pid_list = []
                for elem in buffer_list:
                    buffer_pid_list.extend(list(self.img_pid_dict[elem]))
                    img_count_dict[elem] += 1
                # Check if all pids are unique: bad
                if len(buffer_pid_list) == len(set(buffer_pid_list)):
                    leftover_set.add(buffer_list.pop())
                    buffer_list.append(buffer_list[-1])
                    num_bad_batch += 1
                    #raise Exception
                # Check unique image ids
                else:
                    buffer_img_set = set(buffer_list)
                    if len(buffer_img_set) < len(buffer_list):
                        num_buffer_dup = self.batch_size - len(buffer_img_set)
                        if len(leftover_set) >= num_buffer_dup:
                            buffer_list = list(set(buffer_list))
                            print('==> Num buffer dup: {}'.format(num_buffer_dup))
                            for _ in range(num_buffer_dup):
                                buffer_list.append(leftover_set.pop())
                                num_single_groups += 1
                            assert len(buffer_list) == self.batch_size
                        else:
                            num_dup_batch += 1
                num_tot_batch += 1

                # Count unique IDs
                buffer_img_set = set(buffer_list)
                unique_img_set.update(buffer_img_set)

                # Store the batch
                batch_list.append(buffer_list.copy())

                # Reset the buffer list for next iter
                buffer_list = []
                num_buffer_single = 0

            # Break once we have reached target num batches
            self.self_len = len(self) * self.num_replicas
            if self.split_batch:
                self.self_len = self.self_len // 2
            if num_tot_batch == self.self_len:
                break

        # Improve diversity of batches, include more unique images / pids
        if True:
            sampler_idx_set = set(sampler_idx_list)
            remain_idx_set = sampler_idx_set - unique_img_set
            print('Remaining idx: {}'.format(len(remain_idx_set)))
            new_batch_list = []
            for buffer_list in batch_list:
                new_buffer_list = []
                for img1 in buffer_list:
                    other_pid_set = set() 
                    img_pid_set = self.img_pid_dict[img1]
                    for img2 in buffer_list:
                        if img1 != img2:
                            other_pid_set.update(self.img_pid_dict[img2])
                    if img_pid_set.intersection(other_pid_set):
                        new_buffer_list.append(img1)
                    elif img_count_dict[img1] == 1:
                        new_buffer_list.append(img1)
                    else:
                        new_buffer_list.append(remain_idx_set.pop())
                        #try:
                        #    new_buffer_list.append(remain_idx_set.pop())
                        #except KeyError:
                        #    new_buffer_list = buffer_list
                        #    break
                new_batch_list.append(new_buffer_list)
            batch_list = new_batch_list
            
        # Count stats for final selected batch list
        for buffer_list in batch_list:
            # Count final selected images
            buffer_img_set = set(buffer_list)
            final_unique_img_set.update(buffer_img_set)

            # Count final selected buffer pids
            final_buffer_pid_list = []
            for elem in buffer_list:
                pid_list = list(self.img_pid_dict[elem])
                final_buffer_pid_list.extend(pid_list)
                for pid in pid_list:
                    unique_pid_set.add((elem, pid))

            # Count how many groups of each size there are
            unique_pid_arr, unique_count_arr = np.unique(final_buffer_pid_list, return_counts=True)
            for count in unique_count_arr:
                group_count_dict[count] += 1

        # Count triplets in each batch
        if False:
            # Count stats for final selected batch list
            num_triplet_list = []
            for buffer_list in batch_list:
                # Count final selected buffer pids
                full_label_list = []
                for elem in buffer_list:
                    pid_list = list(self.img_pid_dict[elem])
                    unk_list = list(self.img_unk_dict[elem])
                    full_label_list.extend(pid_list)
                    full_label_list.extend(unk_list)

                # Count the counts
                count_counter = collections.defaultdict(int)
                _, label_count_arr = np.unique(full_label_list, return_counts=True)
                for count in label_count_arr:
                    count_counter[count] += 1

                # Function to compute pairs
                def _compute_pairs(n):
                    return (n*(n-1))//2

                # Determine num single and pair elements
                num_single = 0
                num_pair = 0
                for count, num in count_counter.items():
                    if count == 1:
                        num_single = num
                    else:
                        num_single += (num - 1) * count
                        num_pair += num * _compute_pairs(count)

                # Determine num triplets
                num_triplets = num_single * num_pair
                num_triplet_list.append(num_triplets)
            print('Min/Mean/Max triplets: {}/{}/{}'.format(
                np.min(num_triplet_list), int(np.mean(num_triplet_list)), np.max(num_triplet_list)))

        print('Covered batches: {}/{}'.format(num_tot_batch, self.self_len))
        print('Num leftover IDs: {}'.format(len(leftover_set)))
        print('Num unique ImageID used: {}/{}'.format(len(final_unique_img_set), tot_img))
        print('Num unique PID used: {}/{}'.format(len(unique_pid_set), len(self.pid_set)))
        print('Num singleton groups: {}/{}'.format(num_single_groups, num_tot_groups))
        print('Num duplicate batches: {}/{}'.format(num_dup_batch, num_tot_batch))
        print('Num bad batches: {}/{}'.format(num_bad_batch, num_tot_batch))
        print('Group counts:')
        for count, num in sorted(group_count_dict.items(), key=lambda x: x[0]):
            print('{}: {}'.format(count, num))
        # now we have run out of elements that satisfy
        # the group criteria, let's return the remaining
        # elements so that the size of the sampler is
        # deterministic
        expected_num_batches = self.self_len
        num_remaining = expected_num_batches - num_tot_batch
        #print('Num batches remaining: {}'.format(num_remaining))
        #assert num_remaining == 0

        # Split batches
        if self.split_batch:
            print('==> Splitting batches...')
            np.random.shuffle(batch_list)
            new_batch_list = []
            for i, idx_batch in enumerate(batch_list):
                if (i % 5) == 0:
                    sorted_idx_batch = sorted(idx_batch)
                    for j in range(0, len(sorted_idx_batch), 4):
                        _idx_batch = sorted_idx_batch[j:j+4] 
                        new_batch_list.append(_idx_batch)
                else:
                    new_batch_list.append(idx_batch) 
            batch_list = new_batch_list
            print('==> Done splitting batches.')

        # Batch chaining:
        if False:
            chain_batch_list = [batch_list[-1]+batch_list[0]]
            for batch1, batch2 in zip(batch_list[:-1], batch_list[1:]):
                batch = batch1 + batch2
                chain_batch_list.append(batch)
            batch_list = chain_batch_list
        # Shuffle the batches
        if True:
            np.random.shuffle(batch_list)
        # Get the batches to be used for this process
        print('==> SET THE REPLICA LIST')
        self.replica_list = batch_list[self.rank::self.num_replicas]
        # Shuffle the replica list
        if True:
            np.random.shuffle(self.replica_list)

        # Return the dict mapping idx to batch size
        id_batch_size_dict = {}
        num_idx_orig = 0
        num_idx_repeat = 0
        for idx_batch in self.replica_list:
            batch_len = len(idx_batch)
            for idx in idx_batch:
                image_id = self.image_ids[idx]
                if image_id in id_batch_size_dict:
                    _batch_len = id_batch_size_dict[image_id]
                    id_batch_size_dict[image_id] = max(_batch_len, batch_len)
                    num_idx_repeat += 1
                else:
                    id_batch_size_dict[image_id] = batch_len
                    num_idx_orig += 1
        print('Num idx orig: {}'.format(num_idx_orig))
        print('Num idx repeat: {}'.format(num_idx_repeat))
        return id_batch_size_dict

        # Return the unique pid set
        #return final_unique_img_set

class GroupedPIDBatchSampler4(BatchSampler):
    """
    Wraps another sampler to yield a mini-batch of indices.
    It enforces that the batch only contain elements from the same group.
    It also tries to provide mini-batches which follows an ordering which is
    as close as possible to the ordering from the original sampler.
    Arguments:
        sampler (Sampler): Base sampler.
        group_ids (list[int]): If the sampler produces indices in range [0, N),
            `group_ids` must be a list of `N` ints which contains the group id of each sample.
            The group ids must be a continuous set of integers starting from
            0, i.e. they must be in the range [0, num_groups).
        batch_size (int): Size of mini-batch.
    """
    def __init__(self, sampler, person_ids, image_ids, aspect_ratios_dict, num_pid, img_per_pid, max_single=4,
        num_replicas=None, rank=None, shuffle=True, seed=0, lookup_path=None, split_batch=False):
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

        # Params
        self.sampler = sampler
        self.person_ids = person_ids
        self.image_ids = image_ids
        self.num_pid = num_pid
        self.img_per_pid = img_per_pid
        self.batch_size = self.num_pid * self.img_per_pid
        self.max_single = max_single
        self.split_batch = split_batch
        self.aspect_ratios_dict = aspect_ratios_dict

        # Don't use all of sampler
        #self.sampler = list(self.sampler)[:1000]

        if os.path.exists(lookup_path):
            print('==> Reading lookup file...')
            with open(lookup_path, 'rb') as fp:
                lookup_dict = pickle.load(fp)
                self.img_unk_dict = lookup_dict['img_unk_dict']
                self.pid_set = lookup_dict['pid_set']
                self.img_set_lookup_dict = lookup_dict['img_set_lookup_dict']
                self.img_pid_dict = lookup_dict['img_pid_dict']        
        else:
            print('==> Writing lookup file...')
            ## Compute the master img group set dict
            ## Make groups of pids
            # Compute image, pid lookup dicts
            img_pid_dict = collections.defaultdict(set)
            img_unk_dict = collections.defaultdict(set)
            print('sampler:', self.sampler)
            for img in self.sampler:
                print('img:', img)
                pid_list = self.person_ids[img]
                for pid in pid_list:
                    if pid.startswith('p'):
                        img_pid_dict[img].add(pid)
                    elif pid.startswith('u'):
                        img_unk_dict[img].add(pid)
                    else: raise Exception
            self.img_unk_dict = img_unk_dict

            # Lookup dict for image sets
            self.pid_set = set()
            img_set_lookup_dict = collections.defaultdict(set)
            for img1, pid_set1 in img_pid_dict.items():
                # Count unique img/pid combos
                for pid1 in pid_set1:
                    self.pid_set.add((img1, pid1))    
                #
                for img2 in img_pid_dict:
                    if img2 > img1:
                        pid_set2 = img_pid_dict[img2]
                        if pid_set1.intersection(pid_set2):
                            img_set_lookup_dict[img1].add(img2)
                            img_set_lookup_dict[img2].add(img1)
            for img in img_pid_dict:
                if img not in img_set_lookup_dict:
                    img_set_lookup_dict[img] = set()

            count_dict = collections.defaultdict(int)
            for img, img_set in sorted(img_set_lookup_dict.items(), key=lambda x: len(x[1]), reverse=True):
                count_dict[len(img_set)] += 1
            pprint(count_dict)

            self.img_set_lookup_dict = img_set_lookup_dict
            self.img_pid_dict = img_pid_dict

            # Build lookup dict
            lookup_dict = {
                'img_unk_dict': self.img_unk_dict,
                'pid_set': self.pid_set,
                'img_set_lookup_dict': self.img_set_lookup_dict,
                'img_pid_dict': self.img_pid_dict,
            }

            # Save lookup dict to disk
            with open(lookup_path, 'wb') as fp:
                pickle.dump(lookup_dict, fp)

    def __iter__(self):
        return iter(self.replica_list)

    def __len__(self):
        if self.split_batch:
            return 2 * (len(self.sampler) // self.batch_size) // self.num_replicas
        else:
            return (len(self.sampler) // self.batch_size) // self.num_replicas

    def set_epoch(self, epoch):
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Arguments:
            epoch (int): Epoch number.
        """
        ### Set the epoch
        self.epoch = epoch

        ### Prep the batch and determine its length
        # Set random seed
        np.random.seed(self.seed + self.epoch)

        # Shuffle the index list
        sampler_idx_list = list(self.sampler)
        sampler_idx_arr = np.array(sampler_idx_list)
        sampler_ar_list = [self.aspect_ratios_dict[self.image_ids[i]] for i in sampler_idx_list]
        sampler_ar_arr = np.array(sampler_ar_list)
        sampler_idx_list = [sampler_idx_arr[sampler_ar_arr==0].tolist(), sampler_idx_arr[sampler_ar_arr==1].tolist()]

        # Keep track of stuff
        unique_pid_set = set()
        unique_img_set = set()
        final_unique_img_set = set()
        num_single_groups, num_tot_groups = 0, 0
        num_dup_batch, num_tot_batch = 0, 0
        num_bad_batch = 0
        num_batches = 0
        num_buffer_single = 0
        tot_img = len(self.sampler)
        group_count_dict = collections.defaultdict(int)
        img_count_dict = collections.defaultdict(int)

        # Build the batch list
        batch_list = collections.defaultdict(list)
        leftover_set = collections.defaultdict(set)
        for _sampler_ar, _sampler_idx_list in enumerate(sampler_idx_list): 
            np.random.shuffle(_sampler_idx_list)
            buffer_list = []
            for idx1 in _sampler_idx_list:
                # XXX: this line makes sure aspect ratio matches
                image_id1 = self.image_ids[idx1]
                ar1 = self.aspect_ratios_dict[image_id1]
                if len(buffer_list) > 0:
                    image_id2 = self.image_ids[buffer_list[0]]
                    ar2 = self.aspect_ratios_dict[image_id2]
                    if ar1 != ar2:
                        continue
                else:
                    ar2 = None
                # XXX: this line allowed more unique coverage
                if idx1 in unique_img_set:
                    continue
                img_list = list(self.img_set_lookup_dict[idx1])
                # Put any elements without matching AR in the leftover set
                ar_list = [self.aspect_ratios_dict[self.image_ids[i]] for i in img_list]
                if (len(set(ar_list)) != 1) and (ar2 is not None):
                    img_arr = np.array(img_list)
                    ar_arr = np.array(ar_list)
                    ar_mask = ar_arr == ar2
                    img_list = img_arr[ar_mask].tolist() 
                # Num available to sample
                num_avail = len(img_list)
                # XXX: this line means skipping small groups 
                #if (num_avail + 1) < self.img_per_pid:
                #    continue
                # How many items are left in the batch to fill?
                num_remain = self.batch_size - len(buffer_list)
                # XXX: this line means skipping small groups 
                #if ((num_avail + 1) < self.img_per_pid) and ((num_avail + 1) != num_remain):
                #    continue
                # There should be more than 0 remaining
                assert num_remain > 0
                if num_avail > 0:
                    # Sample img_per_pid by default, otherwise min of num remaining and num available
                    num_sample = min(self.img_per_pid, num_avail+1, num_remain) - 1
                    #print(self.img_per_pid, num_avail+1, num_remain)
                    # Sample from possibilities for this idx
                    idx2_list = np.random.choice(img_list, num_sample, replace=False).tolist()
                    # Use the original idx
                    idx2_list.append(idx1)
                    #buffer_list.extend(idx2_list)
                    ar_list = [self.aspect_ratios_dict[self.image_ids[i]] for i in idx2_list]
                    if len(set(ar_list)) == 1:
                        buffer_list.extend(idx2_list)
                    else:
                        leftover_set[ar1].add(idx1)
                else:
                    if num_buffer_single >= self.max_single:
                        leftover_set[ar1].add(idx1)
                    else:
                        buffer_list.extend([idx1])
                        num_single_groups += 1
                        num_buffer_single += 1
                num_tot_groups += 1

                # Check if batch is ready, and make final changes to it
                if len(buffer_list) == self.batch_size:
                    buffer_pid_list = []
                    for elem in buffer_list:
                        buffer_pid_list.extend(list(self.img_pid_dict[elem]))
                        img_count_dict[elem] += 1
                    # Check if all pids are unique: bad
                    if len(buffer_pid_list) == len(set(buffer_pid_list)):
                        leftover_set.add(buffer_list.pop())
                        buffer_list.append(buffer_list[-1])
                        num_bad_batch += 1
                        raise Exception
                    # Check unique image ids
                    else:
                        buffer_img_set = set(buffer_list)
                        if len(buffer_img_set) < len(buffer_list):
                            num_buffer_dup = self.batch_size - len(buffer_img_set)
                            if len(leftover_set[ar2]) >= num_buffer_dup:
                                buffer_list = list(set(buffer_list))
                                print('==> Num buffer dup: {}'.format(num_buffer_dup))
                                for _ in range(num_buffer_dup):
                                    buffer_list.append(leftover_set[ar2].pop())
                                    num_single_groups += 1
                                assert len(buffer_list) == self.batch_size
                            else:
                                num_dup_batch += 1
                    num_tot_batch += 1

                    # Count unique IDs
                    buffer_img_set = set(buffer_list)
                    unique_img_set.update(buffer_img_set)

                    # Store the batch
                    batch_list[self.aspect_ratios_dict[self.image_ids[buffer_list[0]]]].append(buffer_list.copy())

                    # Reset the buffer list for next iter
                    buffer_list = []
                    num_buffer_single = 0

                # Break once we have reached target num batches
                self.self_len = len(self) * self.num_replicas
                if self.split_batch:
                    self.self_len = self.self_len // 2
                if num_tot_batch == self.self_len:
                    break

            # Improve diversity of batches, include more unique images / pids
            if True:
                sampler_idx_set = set(_sampler_idx_list)
                remain_idx_list = list(sampler_idx_set - unique_img_set)
                print('Remaining idx: {}'.format(len(remain_idx_list)))
                new_batch_list = []
                for buffer_list in batch_list[_sampler_ar]:
                    new_buffer_list = []
                    for img1 in buffer_list:
                        other_pid_set = set() 
                        img_pid_set = self.img_pid_dict[img1]
                        for img2 in buffer_list:
                            if img1 != img2:
                                other_pid_set.update(self.img_pid_dict[img2])
                        if img_pid_set.intersection(other_pid_set):
                            new_buffer_list.append(img1)
                        elif img_count_dict[img1] == 1:
                            new_buffer_list.append(img1)
                        elif len(remain_idx_list) == 0:
                            new_buffer_list.append(img1)
                        else:
                            new_buffer_list.append(remain_idx_list.pop())
                            #try:
                            #    new_buffer_list.append(remain_idx_set.pop())
                            #except KeyError:
                            #    new_buffer_list = buffer_list
                            #    break
                    new_batch_list.append(new_buffer_list)
                batch_list[_sampler_ar] = new_batch_list

        # Make sure AR matches for all elems per batch
        batch_list = batch_list[0] + batch_list[1]
        for buffer_list in batch_list:
            ar_list = [self.aspect_ratios_dict[self.image_ids[i]] for i in buffer_list]
            assert len(set(ar_list)) == 1
            
        # Count stats for final selected batch list
        for buffer_list in batch_list:
            # Count final selected images
            buffer_img_set = set(buffer_list)
            final_unique_img_set.update(buffer_img_set)

            # Count final selected buffer pids
            final_buffer_pid_list = []
            for elem in buffer_list:
                pid_list = list(self.img_pid_dict[elem])
                final_buffer_pid_list.extend(pid_list)
                for pid in pid_list:
                    unique_pid_set.add((elem, pid))

            # Count how many groups of each size there are
            unique_pid_arr, unique_count_arr = np.unique(final_buffer_pid_list, return_counts=True)
            for count in unique_count_arr:
                group_count_dict[count] += 1

        # Count triplets in each batch
        if False:
            # Count stats for final selected batch list
            num_triplet_list = []
            for buffer_list in batch_list:
                # Count final selected buffer pids
                full_label_list = []
                for elem in buffer_list:
                    pid_list = list(self.img_pid_dict[elem])
                    unk_list = list(self.img_unk_dict[elem])
                    full_label_list.extend(pid_list)
                    full_label_list.extend(unk_list)

                # Count the counts
                count_counter = collections.defaultdict(int)
                _, label_count_arr = np.unique(full_label_list, return_counts=True)
                for count in label_count_arr:
                    count_counter[count] += 1

                # Function to compute pairs
                def _compute_pairs(n):
                    return (n*(n-1))//2

                # Determine num single and pair elements
                num_single = 0
                num_pair = 0
                for count, num in count_counter.items():
                    if count == 1:
                        num_single = num
                    else:
                        num_single += (num - 1) * count
                        num_pair += num * _compute_pairs(count)

                # Determine num triplets
                num_triplets = num_single * num_pair
                num_triplet_list.append(num_triplets)
            print('Min/Mean/Max triplets: {}/{}/{}'.format(
                np.min(num_triplet_list), int(np.mean(num_triplet_list)), np.max(num_triplet_list)))

        print('Covered batches: {}/{}'.format(num_tot_batch, self.self_len))
        print('Num leftover IDs: {}'.format(len(leftover_set[0])+len(leftover_set[1])))
        print('Num unique ImageID used: {}/{}'.format(len(final_unique_img_set), tot_img))
        print('Num unique PID used: {}/{}'.format(len(unique_pid_set), len(self.pid_set)))
        print('Num singleton groups: {}/{}'.format(num_single_groups, num_tot_groups))
        print('Num duplicate batches: {}/{}'.format(num_dup_batch, num_tot_batch))
        print('Num bad batches: {}/{}'.format(num_bad_batch, num_tot_batch))
        print('Group counts:')
        for count, num in sorted(group_count_dict.items(), key=lambda x: x[0]):
            print('{}: {}'.format(count, num))
        # now we have run out of elements that satisfy
        # the group criteria, let's return the remaining
        # elements so that the size of the sampler is
        # deterministic
        expected_num_batches = self.self_len
        num_remaining = expected_num_batches - num_tot_batch
        #print('Num batches remaining: {}'.format(num_remaining))
        #assert num_remaining == 0

        # Split batches
        if self.split_batch:
            print('==> Splitting batches...')
            np.random.shuffle(batch_list)
            new_batch_list = []
            for i, idx_batch in enumerate(batch_list):
                if (i % 5) == 0:
                    sorted_idx_batch = sorted(idx_batch)
                    for j in range(0, len(sorted_idx_batch), 4):
                        _idx_batch = sorted_idx_batch[j:j+4] 
                        new_batch_list.append(_idx_batch)
                else:
                    new_batch_list.append(idx_batch) 
            batch_list = new_batch_list
            print('==> Done splitting batches.')

        # Batch chaining:
        if False:
            chain_batch_list = [batch_list[-1]+batch_list[0]]
            for batch1, batch2 in zip(batch_list[:-1], batch_list[1:]):
                batch = batch1 + batch2
                chain_batch_list.append(batch)
            batch_list = chain_batch_list
        # Shuffle the batches
        if True:
            np.random.shuffle(batch_list)
        # Get the batches to be used for this process
        print('==> SET THE REPLICA LIST')
        self.replica_list = batch_list[self.rank::self.num_replicas]
        # Shuffle the replica list
        if True:
            np.random.shuffle(self.replica_list)

        # Return the dict mapping idx to batch size
        id_batch_size_dict = {}
        num_idx_orig = 0
        num_idx_repeat = 0
        for idx_batch in self.replica_list:
            batch_len = len(idx_batch)
            for idx in idx_batch:
                image_id = self.image_ids[idx]
                if image_id in id_batch_size_dict:
                    _batch_len = id_batch_size_dict[image_id]
                    id_batch_size_dict[image_id] = max(_batch_len, batch_len)
                    num_idx_repeat += 1
                else:
                    id_batch_size_dict[image_id] = batch_len
                    num_idx_orig += 1
        print('Num idx orig: {}'.format(num_idx_orig))
        print('Num idx repeat: {}'.format(num_idx_repeat))
        return id_batch_size_dict

        # Return the unique pid set
        #return final_unique_img_set

def find_connected_components(img_id_set_dict):
    n = len(img_id_set_dict)
    adj_mat = np.zeros((n, n), dtype=np.uint8)
    for idx_i, (img_id_i, img_pid_set_i) in tqdm(enumerate(img_id_set_dict.items()), total=n):
        for idx_j, (img_id_j, img_pid_set_j) in enumerate(img_id_set_dict.items()):
            if img_id_i != img_id_j:
                if len(img_pid_set_i.intersection(img_pid_set_j)) > 0:
                    adj_mat[idx_i, idx_j] = 1
    G = nx.convert_matrix.from_numpy_matrix(adj_mat)
    C = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    return C

def random_edge_cover(g, seed=0):
    # Seed RNG
    random.seed(seed)
    edge_list = [] 
    unused_node_list = list(g.nodes())
    random.shuffle(unused_node_list)
    used_node_list = []
    cover_set = set()
    # While there are still nodes explore
    while len(unused_node_list) > 0:
        # Randomly choose a vertex
        n1 = unused_node_list.pop()
        used_node_list.append(n1)
        # Get list of edges in g which contain n
        edge_list = list(g.edges(n1))
        random.shuffle(edge_list)
        ## Randomly choose an edge
        # First try to pick an edge where the other vertex has not been picked
        for e in edge_list:
            n2 = e[1]
            if n2 not in used_node_list:
                unused_node_list.remove(n2)
                used_node_list.append(n2)
                cover_set.add(e)
                break 
        # If no such edge exists, pick an edge where the other node has minimal degree
        # in the subgraph induced by the current cover set
        else:
            ge = nx.Graph(cover_set)
            n2_list = list(ge.nodes())
            random.shuffle(n2_list)
            d2_list = []
            for n2 in n2_list:
                d2_list.append(ge.degree(n2))
            assert len(n2_list) == len(d2_list)
            # Sorted from least to greatest degree
            for n2, d2 in sorted(zip(n2_list, d2_list), key=lambda x: x[1]):
                e = (n1, n2)
                cover_set.add(e)
                break
                     
    # Make sure this is an edge cover of the original graph
    assert nx.is_edge_cover(g, cover_set)

    # Return cover set
    return cover_set

# Function to find batches of 2 elements
def find_batches(C, idx, m, w, seed=0):
    # Number of pairs must be divisible by this
    # If it is not, randomly remove singleton pairs
    d = m * w

    # Get list of image pairs
    pair_list = []
    num_list = []
    c_counter = collections.Counter()
    for c_idx, c in enumerate(C):
        # Convert components with one element to a batch with repeated index
        if len(c.nodes) == 1:
            _cover_list = [(c_idx, tuple(2*list(c.nodes())))]
        # If a component is just an edge, use the edge
        elif len(c.nodes) == 2:
            _cover_list = [(c_idx, list(c.edges())[0])]
        # If a component has a least 2 edges, find a random edge cover
        else:
            _cover_list = [(c_idx, cover) for cover in random_edge_cover(c, seed=seed)]
        # Convert cover list to covert list using correct dataset indices
        cover_list = [(c_idx, (idx[t[0]], idx[t[1]])) for _, t in _cover_list]
        # Convert cover set back to img_ids
        pair_list.extend(cover_list)
        num_list.extend([len(c.nodes)]*len(cover_list))
        c_counter[c_idx] = len(c.nodes)
    pair_arr = np.array(pair_list, dtype=object)

    # Pare pairs
    r = len(pair_arr) % d
    num_arr = np.array(num_list)
    singleton_mask = num_arr == 1
    if r != 0:
        multi_pair_arr = pair_arr[~singleton_mask]
        single_pair_arr = pair_arr[singleton_mask]
        if len(single_pair_arr) > r:
            num_keep = len(single_pair_arr) - r
            num_multi_keep = len(multi_pair_arr)
            rand_single_idx = np.random.choice(range(len(single_pair_arr)), num_keep, replace=False)
            new_single_pair_arr = single_pair_arr[rand_single_idx]
            new_pair_arr = np.concatenate([new_single_pair_arr, multi_pair_arr], axis=0)
        else:
            r = len(multi_pair_arr) % d
            num_keep = 0
            num_multi_keep = len(multi_pair_arr) - r
            rand_multi_idx = np.random.choice(range(len(multi_pair_arr)), num_multi_keep, replace=False)
            new_multi_pair_arr = multi_pair_arr[rand_multi_idx]
            new_pair_arr = new_multi_pair_arr
        pair_arr = new_pair_arr
        num_single, num_multi = num_keep, num_multi_keep
    else:
        num_single, num_multi = singleton_mask.sum().item(), (~singleton_mask).sum().item()
    assert (len(pair_arr) % d) == 0

    # Print stats
    print('Num single/multi pair: {}/{}'.format(num_single, num_multi))

    # Form dict of pairs
    c_idx_arr = np.array([pair[0] for pair in pair_arr])
    elem_arr = np.array([pair[1] for pair in pair_arr])
    pair_dict = {}
    for c_idx in c_counter:
        pair_dict[c_idx] = elem_arr[c_idx_arr==c_idx].tolist()
        
    # Form batches from image pairs by putting pairs from different components together
    num_batches = len(pair_arr) // m
    batch_list = []
    num_repeat, num_unique = 0, 0
    while len(pair_dict) > 0:
        # Get list of current keys
        keys = list(pair_dict.keys())

        # Make sure the desired sampling is possible
        if len(keys) < m:
            num_repeat += 1
            while True:
                sampled_keys = np.random.choice(keys, m, replace=True) 
                s_counter = collections.Counter(sampled_keys)
                for c_idx in s_counter:
                    if len(pair_dict[c_idx]) < s_counter[c_idx]:
                        break
                else:
                    break
        else: 
            # Randomly sample m keys
            num_unique += 1
            sampled_keys = np.random.choice(keys, m, replace=False) 

        # Create the batch
        batch = []
        for key in sampled_keys:
            pair_list = pair_dict[key]
            if len(pair_list) > 0:
                pair_idx = np.random.choice(range(len(pair_list)))
                batch.append(pair_list[pair_idx])
                del pair_dict[key][pair_idx]
            else:
                print('WARNING: Bad batch! (find_batches)')
                return None

        # Store batch
        batch_list.append(batch)
            
        # Delete keys with no more elements
        for key in list(pair_dict.keys()):
            if len(pair_dict[key]) == 0:
                del pair_dict[key]

    # Form final batch_list from batch_list
    final_batch_list = [sum(batch, []) for batch in batch_list]

    # Print stats
    print('Num unique/repeat batch: {}/{}'.format(num_unique, num_repeat))

    # Return batch list
    return final_batch_list
    

class GroupedPIDBatchSamplerEdgeCover(BatchSampler):
    """
    Wraps another sampler to yield a mini-batch of indices.
    It enforces that the batch only contain elements from the same group.
    It also tries to provide mini-batches which follows an ordering which is
    as close as possible to the ordering from the original sampler.
    Arguments:
        sampler (Sampler): Base sampler.
        group_ids (list[int]): If the sampler produces indices in range [0, N),
            `group_ids` must be a list of `N` ints which contains the group id of each sample.
            The group ids must be a continuous set of integers starting from
            0, i.e. they must be in the range [0, num_groups).
        batch_size (int): Size of mini-batch.
    """
    def __init__(self, sampler, person_ids, image_ids, is_known,
            num_pid, img_per_pid, aspect_ratios_dict=None,
            max_single=4, num_replicas=None, rank=None, shuffle=True,
            seed=0, lookup_path=None, ignore_unk=True):

        #
        if not isinstance(sampler, Sampler):
            raise ValueError(
                "sampler should be an instance of "
                "torch.utils.data.Sampler, but got sampler={}".format(sampler)
            )

        # This sampler only works for case with 2 images per pid
        assert img_per_pid == 2 

        # Distributed params
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0

        # Params
        self.sampler = sampler
        self.person_ids = person_ids
        self.is_known = is_known
        self.pid_set = set(sum(person_ids, []))
        self.num_pid = num_pid
        self.img_per_pid = img_per_pid
        self.batch_size = self.num_pid * self.img_per_pid
        self.max_single = max_single

        # Load adjacenct matrix
        if (lookup_path is not None) and os.path.exists(lookup_path):
            print('==> Loading existing adj mat dict...')
            with open(lookup_path, 'rb') as fp:
                connected_components = pickle.load(fp)
        else:
            print('==> Writing adj mat dict: ignore_unk={}'.format(ignore_unk))
            img_idx_set_dict = collections.defaultdict(lambda : collections.defaultdict(set))
            if aspect_ratios_dict is None:
                for i in self.sampler:
                    image_id = image_ids[i]
                    ar = 1
                    _pid_list = self.person_ids[i]
                    _unk_list = self.is_known[i]
                    if ignore_unk:
                        pid_list = [p for u, p in zip(_unk_list, _pid_list) if u]
                    else:
                        pid_list = _pid_list
                    img_idx_set_dict[ar][i] = set(pid_list)
            else:
                for i in self.sampler:
                    image_id = image_ids[i]
                    ar = aspect_ratios_dict[image_id]
                    _pid_list = self.person_ids[i]
                    _unk_list = self.is_known[i]
                    if ignore_unk:
                        pid_list = [p for u, p in zip(_unk_list, _pid_list) if u]
                    else:
                        pid_list = _pid_list
                    img_idx_set_dict[ar][i] = set(pid_list)
            connected_components = {}
            for ar in img_idx_set_dict:
                connected_components[ar] = find_connected_components(img_idx_set_dict[ar]), list(img_idx_set_dict[ar].keys())
            if lookup_path is not None:
                with open(lookup_path, 'wb') as fp:
                    pickle.dump(connected_components, fp)
        self.connected_components = connected_components

        # set epoch once so the replica_list has a length
        self.set_epoch(0)

    def __iter__(self):
        return iter(self.replica_list)

    def __len__(self):
        #return (len(self.sampler) // self.batch_size) // self.num_replicas
        return len(self.replica_list)

    def set_epoch(self, epoch):
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Arguments:
            epoch (int): Epoch number.
        """
        ### Set the epoch
        self.epoch = epoch

        ### Prep the batch and determine its length
        # Set seed
        seed = self.seed + self.epoch

        # Set random seed
        np.random.seed(seed)

        # Find min edge cover
        batch_list = []
        for cc, idx in self.connected_components.values():
            batch = None
            while batch == None:
                batch = find_batches(cc, idx, self.num_pid, self.num_replicas, seed=seed)
                if batch == None:
                    self.seed += 1
                    seed = self.seed + self.epoch
                else:
                    batch_list += batch

        # Count stats for final selected batch list
        final_unique_img_set = set()
        unique_pid_set = set()
        for buffer_list in batch_list:
            # Count final selected images
            buffer_img_set = set(buffer_list)
            final_unique_img_set.update(buffer_img_set)

            # Count final selected buffer pids
            final_buffer_pid_list = []
            for elem in buffer_list:
                pid_list = list(self.person_ids[elem])
                final_buffer_pid_list.extend(pid_list)
                for pid in pid_list:
                    unique_pid_set.add((elem, pid))

        print('Num batches: {}'.format(len(batch_list)))
        print('Num unique ImageID used: {}/{}'.format(len(final_unique_img_set), len(self.person_ids)))
        print('Num unique PID used: {}/{}'.format(len(unique_pid_set), len(self.pid_set)))
        tot_idx_list = sum(batch_list, [])
        num_idx_orig = len(set(tot_idx_list))
        num_idx_repeat = len(tot_idx_list) - num_idx_orig
        print('Num idx orig: {}'.format(num_idx_orig))
        print('Num idx repeat: {}'.format(num_idx_repeat))

        # Shuffle the batches
        np.random.shuffle(batch_list)
        # Get the batches to be used for this process
        print('==> SET THE REPLICA LIST')
        self.replica_list = batch_list[self.rank::self.num_replicas]
        # Shuffle the replica list
        np.random.shuffle(self.replica_list)

        # Return the unique pid set
        return final_unique_img_set

def _compute_aspect_ratios_slow(dataset, indices=None):
    print("Your dataset doesn't support the fast path for "
          "computing the aspect ratios, so will iterate over "
          "the full dataset and load every image instead. "
          "This might take some time...")
    if indices is None:
        indices = range(len(dataset))

    class SubsetSampler(Sampler):
        def __init__(self, indices):
            self.indices = indices

        def __iter__(self):
            return iter(self.indices)

        def __len__(self):
            return len(self.indices)

    sampler = SubsetSampler(indices)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, sampler=sampler,
        num_workers=14,  # you might want to increase it for faster processing
        collate_fn=lambda x: x[0])
    aspect_ratios = []
    with tqdm(total=len(dataset)) as pbar:
        for _i, (img, _) in enumerate(data_loader):
            pbar.update(1)
            height, width = img.shape[-2:]
            aspect_ratio = float(width) / float(height)
            aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def _compute_aspect_ratios_custom_dataset(dataset, indices=None):
    if indices is None:
        indices = range(len(dataset))
    aspect_ratios = []
    for i in indices:
        height, width = dataset.get_height_and_width(i)
        aspect_ratio = float(width) / float(height)
        aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def _compute_aspect_ratios_coco_dataset(dataset, indices=None):
    if indices is None:
        indices = range(len(dataset))
    aspect_ratios = []
    for i in indices:
        img_info = dataset.coco.imgs[dataset.ids[i]]
        aspect_ratio = float(img_info["width"]) / float(img_info["height"])
        aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def _compute_aspect_ratios_voc_dataset(dataset, indices=None):
    if indices is None:
        indices = range(len(dataset))
    aspect_ratios = []
    for i in indices:
        # this doesn't load the data into memory, because PIL loads it lazily
        width, height = Image.open(dataset.images[i]).size
        aspect_ratio = float(width) / float(height)
        aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def _compute_aspect_ratios_subset_dataset(dataset, indices=None):
    if indices is None:
        indices = range(len(dataset))

    ds_indices = [dataset.indices[i] for i in indices]
    return compute_aspect_ratios(dataset.dataset, ds_indices)


def compute_aspect_ratios(dataset, indices=None):
    if hasattr(dataset, "get_height_and_width"):
        return _compute_aspect_ratios_custom_dataset(dataset, indices)

    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return _compute_aspect_ratios_coco_dataset(dataset, indices)

    if isinstance(dataset, torchvision.datasets.VOCDetection):
        return _compute_aspect_ratios_voc_dataset(dataset, indices)

    if isinstance(dataset, torch.utils.data.Subset):
        return _compute_aspect_ratios_subset_dataset(dataset, indices)

    # slow path
    return _compute_aspect_ratios_slow(dataset, indices)

def _get_pids_coco_dataset_old(dataset, indices=None):
    if indices is None:
        indices = range(len(dataset))
    pids = []
    img_pid_dict = defaultdict(list)
    #for i in range(len(dataset.coco.anns)):
    for i in dataset.coco.anns:
        ann = dataset.coco.anns[i]
        img_pid_dict[ann['image_id']].append(ann['person_id']) 
    for i in indices:
        pid_list = img_pid_dict[dataset.ids[i]]
        pids.append(pid_list)
    return pids

def _get_pids_coco_dataset(dataset, indices=None):
    if indices is None:
        indices = range(len(dataset))
    #print('Idx len/max:', len(indices), max(indices))
    pids, ids, unks = [], [], []
    img_pid_dict = defaultdict(list)
    for i in dataset.coco.anns:
        ann = dataset.coco.anns[i]
        img_pid_dict[ann['image_id']].append((ann['person_id'], ann['is_known'])) 
    for i in indices:
        ids.append(dataset.ids[i])
        pid_list, unk_list = list(zip(*img_pid_dict[dataset.ids[i]]))
        pids.append(list(pid_list))
        unks.append(list(unk_list))
    return ids, pids, unks

def _get_pids_concat_dataset(concat_dataset, indices=None):
    pids, ids, unks = [], [], []
    for dataset in concat_dataset.datasets:
        if indices is None:
            indices = range(len(dataset))
        img_pid_dict = defaultdict(list)
        for i in dataset.coco.anns:
            ann = dataset.coco.anns[i]
            img_pid_dict[ann['image_id']].append((ann['person_id'], ann['is_known'])) 
        for i in indices:
            ids.append(dataset.ids[i])
            pid_list, unk_list = list(zip(*img_pid_dict[dataset.ids[i]]))
            pids.append(list(pid_list))
            unks.append(list(unk_list))
    return ids, pids, unks

def _get_pids_subset_dataset(dataset, indices=None):
    if indices is None:
        indices = range(len(dataset))

    ds_indices = [dataset.indices[i] for i in indices]
    return get_pids(dataset.dataset, ds_indices)

def get_pids(dataset, indices=None):
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return _get_pids_coco_dataset(dataset, indices)
    elif isinstance(dataset, torch.utils.data.ConcatDataset):
        return _get_pids_concat_dataset(dataset, indices)
    elif isinstance(dataset, torch.utils.data.Subset):
        return _get_pids_subset_dataset(dataset, indices)

def _quantize(x, bins):
    bins = copy.deepcopy(bins)
    bins = sorted(bins)
    quantized = list(map(lambda y: bisect.bisect_right(bins, y), x))
    return quantized


def create_pid_groups(dataset, k=0):
    return get_pids(dataset)

if __name__=='__main__':
    # Imports
    from osr.utils import ndl_utils
    from osr.ndl import dist_utils

    # Variables
    workers = 4
    pid_per_batch = 4
    img_per_pid = 4
    num_replicas = 4
    dataset_name = 'cuhk'
    batch_comp = (pid_per_batch, img_per_pid)

    # Dataset
    lookup_path = './{0}_lookup_{1}x{2}.pkl'.format(dataset_name, *batch_comp)
    dataset = ndl_utils.get_coco('/datasets/cuhk', dataset_name=dataset_name, image_set='train', transforms=None)
    train_sampler = torch.utils.data.SequentialSampler(dataset)
    person_ids, image_ids = create_pid_groups(dataset)

    # Sampler
    sampler_list, loader_list = [], []
    unique_pid_set_list = []
    for rank in range(num_replicas):
        train_batch_sampler = GroupedPIDBatchSampler3(train_sampler, person_ids,
            num_pid=pid_per_batch, img_per_pid=img_per_pid,
            num_replicas=4, rank=rank, shuffle=True, seed=0,
            lookup_path=lookup_path)
        sampler_list.append(train_batch_sampler)

        data_loader = torch.utils.data.DataLoader(
            dataset, batch_sampler=sampler_list[rank], num_workers=workers,
            collate_fn=dist_utils.collate_fn)
        loader_list.append(data_loader)
        unique_pid_set_list.append(set())

    # Test run
    num_epochs = 10
    for epoch in range(num_epochs):
        print('Epoch: {}'.format(epoch))
        for i in range(num_replicas):
            #print('\tReplica: {}'.format(i))
            unique_pid_set_list[i].update(sampler_list[i].set_epoch(epoch))
            print('\tPID set coverage {}: {}'.format(i, len(unique_pid_set_list[i])))
        #for _ in loader_list[i]:
        #    pass
        #exit()
