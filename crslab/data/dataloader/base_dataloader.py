# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/23, 2020/12/7
# @Author : Kun Zhou, Xiaolei Wang
# @Email  : francis_kun_zhou@163.com, wxl1999@foxmail.com

import random
from abc import ABC
from math import ceil

from loguru import logger
from tqdm import tqdm


class BaseDataLoader(ABC):
    def __init__(self, opt, dataset):
        self.opt = opt
        self.dataset = dataset
        self.scale = opt.get('scale', 1)
        assert 0 < self.scale <= 1

    def get_data(self, batch_fn, batch_size, shuffle=True, process_fn=None):
        """collate batch data for system to fit

        Args:
            batch_fn (func): function to collate data
            batch_size (int):
            shuffle (bool, optional): Defaults to True.
            process_fn (func, optional): function to process dataset before batchify. Defaults to None.

        Yields:
            tuple or dict of torch.Tensor: batch data for system to fit
        """
        dataset = self.dataset
        if process_fn is not None:
            dataset = process_fn()
            logger.info('[Finish dataset process before batchify]')
        dataset = dataset[:ceil(len(dataset) * self.scale)]
        logger.debug(f'[Dataset size: {len(dataset)}]')

        batch_num = ceil(len(dataset) / batch_size)
        idx_list = list(range(len(dataset)))
        if shuffle:
            random.shuffle(idx_list)

        for start_idx in tqdm(range(batch_num)):
            batch_idx = idx_list[start_idx * batch_size: (start_idx + 1) * batch_size]
            batch = [dataset[idx] for idx in batch_idx]
            yield batch_fn(batch)

    def get_conv_data(self, batch_size, shuffle=True):
        return self.get_data(self.conv_batchify, batch_size, shuffle, self.conv_process_fn)

    def get_rec_data(self, batch_size, shuffle=True):
        return self.get_data(self.rec_batchify, batch_size, shuffle, self.rec_process_fn)

    def get_policy_data(self, batch_size, shuffle=True):
        return self.get_data(self.policy_batchify, batch_size, shuffle, self.policy_process_fn)

    def conv_process_fn(self):
        return self.dataset

    def conv_batchify(self, batch):
        raise NotImplementedError('dataloader must implement conv_batchify() method')

    def rec_process_fn(self):
        return self.dataset

    def rec_batchify(self, batch):
        raise NotImplementedError('dataloader must implement rec_batchify() method')

    def policy_process_fn(self):
        return self.dataset

    def policy_batchify(self, batch):
        raise NotImplementedError('dataloader must implement policy_batchify() method')

    def retain_recommender_target(self):
        dataset = []
        for conv_dict in tqdm(self.dataset):
            if conv_dict['role'] == 'Recommender':
                dataset.append(conv_dict)
        return dataset
