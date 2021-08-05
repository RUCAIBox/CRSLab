# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/23, 2020/12/29, 2021/8/5
# @Author : Kun Zhou, Xiaolei Wang, Chenzhan Shang
# @Email  : francis_kun_zhou@163.com, wxl1999@foxmail.com, czshang@outlook.com

import random
from abc import ABC

from loguru import logger
from math import ceil
from tqdm import tqdm


class SupervisedAgent(ABC):
    """Abstract class of supervised agent

    Notes:
        ``'scale'`` can be set in config to limit the size of dataset.

    """

    def __init__(self, opt, dataset):
        """
        Args:
            opt (Config or dict): config for supervised or the whole system.
            dataset: dataset

        """
        self.opt = opt
        self.data = {
            'train': dataset.train_data,
            'valid': dataset.valid_data,
            'test': dataset.test_data,
        }
        self.other_data = dataset.other_data
        self.scale = opt.get('scale', 1)
        assert 0 < self.scale <= 1

    def get_data(self, mode, batch_fn, batch_size, shuffle=True, process_fn=None):
        """Collate batch data for system to fit

        Args:
            mode (string): train, valid or test.
            batch_fn (func): function to collate data
            batch_size (int):
            shuffle (bool, optional): Defaults to True.
            process_fn (func, optional): function to process dataset before batchify. Defaults to None.

        Yields:
            tuple or dict of torch.Tensor: batch data for system to fit

        """
        assert mode in ('train', 'valid', 'test')
        data = self.data[mode]
        if process_fn is not None:
            data = process_fn(mode)
            logger.info('[Finish dataset process before batchify]')
        data = data[:ceil(len(data) * self.scale)]
        logger.debug(f'[Dataset size: {len(data)}]')

        batch_num = ceil(len(data) / batch_size)
        idx_list = list(range(len(data)))
        if shuffle:
            random.shuffle(idx_list)

        for start_idx in tqdm(range(batch_num)):
            batch_idx = idx_list[start_idx * batch_size: (start_idx + 1) * batch_size]
            batch = [data[idx] for idx in batch_idx]
            yield batch_fn(batch)

    def get_conv_data(self, mode, batch_size, shuffle=True):
        """get_data wrapper for conversation.

        You can implement your own process_fn in ``conv_process_fn``, batch_fn in ``conv_batchify``.

        Args:
            mode (string): train, valid or test.
            batch_size (int):
            shuffle (bool, optional): Defaults to True.

        Yields:
            tuple or dict of torch.Tensor: batch data for conversation.

        """
        return self.get_data(mode, self.conv_batchify, batch_size, shuffle, self.conv_process_fn)

    def get_rec_data(self, mode, batch_size, shuffle=True):
        """get_data wrapper for recommendation.

        You can implement your own process_fn in ``rec_process_fn``, batch_fn in ``rec_batchify``.

        Args:
            mode (string): train, valid or test.
            batch_size (int):
            shuffle (bool, optional): Defaults to True.

        Yields:
            tuple or dict of torch.Tensor: batch data for recommendation.

        """
        return self.get_data(mode, self.rec_batchify, batch_size, shuffle, self.rec_process_fn)

    def get_policy_data(self, mode, batch_size, shuffle=True):
        """get_data wrapper for policy.

        You can implement your own process_fn in ``self.policy_process_fn``, batch_fn in ``policy_batchify``.

        Args:
            mode (string): train, valid or test.
            batch_size (int):
            shuffle (bool, optional): Defaults to True.

        Yields:
            tuple or dict of torch.Tensor: batch data for policy.

        """
        return self.get_data(mode, self.policy_batchify, batch_size, shuffle, self.policy_process_fn)

    def conv_process_fn(self, mode):
        """Process whole data for conversation before batch_fn.

        Returns:
            processed dataset. Defaults to return the same as `self.dataset`.

        """
        return self.data[mode]

    def conv_batchify(self, batch):
        """batchify data for conversation after process.

        Args:
            batch (list): processed batch dataset.

        Returns:
            batch data for the system to train conversation part.
        """
        raise NotImplementedError('supervised must implement conv_batchify() method')

    def rec_process_fn(self, mode):
        """Process whole data for recommendation before batch_fn.

        Returns:
            processed dataset. Defaults to return the same as `self.dataset`.

        """
        return self.data[mode]

    def rec_batchify(self, batch):
        """batchify data for recommendation after process.

        Args:
            batch (list): processed batch dataset.

        Returns:
            batch data for the system to train recommendation part.
        """
        raise NotImplementedError('supervised must implement rec_batchify() method')

    def policy_process_fn(self, mode):
        """Process whole data for policy before batch_fn.

        Returns:
            processed dataset. Defaults to return the same as `self.dataset`.

        """
        return self.data[mode]

    def policy_batchify(self, batch):
        """batchify data for policy after process.

        Args:
            batch (list): processed batch dataset.

        Returns:
            batch data for the system to train policy part.
        """
        raise NotImplementedError('supervised must implement policy_batchify() method')

    def retain_recommender_target(self, mode):
        """keep data whose role is recommender.

        Returns:
            Recommender part of ``self.dataset``.

        """
        dataset = []
        for conv_dict in tqdm(self.data[mode]):
            if conv_dict['role'] == 'Recommender':
                dataset.append(conv_dict)
        return dataset

    def rec_interact(self, data):
        """process user input data for system to recommend.

        Args:
            data: user input data.

        Returns:
            data for system to recommend.
        """
        pass

    def conv_interact(self, data):
        """Process user input data for system to converse.

        Args:
            data: user input data.

        Returns:
            data for system in converse.
        """
        pass
