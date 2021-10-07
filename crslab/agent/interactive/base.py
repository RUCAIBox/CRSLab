# @Time   : 2021/8/4
# @Author : Chenzhan Shang
# @Email  : czshang@outlook.com

import random
from abc import ABC, abstractmethod

from loguru import logger
from math import ceil
from tqdm import tqdm
from crslab.utils import AgentType


class InteractiveAgent(ABC):
    """Abstract class of interactive agent

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
        self.dataset = dataset
        self.scale = opt.get('scale', 1)
        assert 0 < self.scale <= 1
        self._agent_type = self._set_agent_type()
        assert isinstance(self._agent_type, AgentType)
        if self._agent_type.value != self.dataset.dataset_type:
            raise TypeError(f"Dataset type '{self.dataset.dataset_type.name}' does not correspond to Agent type" +
                            f"'{self._agent_type.name}'.")

    @property
    def agent_type(self):
        return self._agent_type

    @abstractmethod
    def _set_agent_type(self) -> AgentType:
        pass

    def get_data(self, batch_fn, batch_size, shuffle=True, process_fn=None):
        """Collate batch data for system to fit

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
        """get_data wrapper for conversation.

        You can implement your own process_fn in ``conv_process_fn``, batch_fn in ``conv_batchify``.

        Args:
            batch_size (int):
            shuffle (bool, optional): Defaults to True.

        Yields:
            tuple or dict of torch.Tensor: batch data for conversation.

        """
        return self.get_data(self.conv_batchify, batch_size, shuffle, self.conv_process_fn)

    def get_rec_data(self, batch_size, shuffle=True):
        """get_data wrapper for recommendation.

        You can implement your own process_fn in ``rec_process_fn``, batch_fn in ``rec_batchify``.

        Args:
            batch_size (int):
            shuffle (bool, optional): Defaults to True.

        Yields:
            tuple or dict of torch.Tensor: batch data for recommendation.

        """
        return self.get_data(self.rec_batchify, batch_size, shuffle, self.rec_process_fn)

    def get_policy_data(self, batch_size, shuffle=True):
        """get_data wrapper for policy.

        You can implement your own process_fn in ``self.policy_process_fn``, batch_fn in ``policy_batchify``.

        Args:
            batch_size (int):
            shuffle (bool, optional): Defaults to True.

        Yields:
            tuple or dict of torch.Tensor: batch data for policy.

        """
        return self.get_data(self.policy_batchify, batch_size, shuffle, self.policy_process_fn)

    def conv_process_fn(self):
        """Process whole data for conversation before batch_fn.

        Returns:
            processed dataset. Defaults to return the same as `self.dataset`.

        """
        return self.dataset

    def conv_batchify(self, batch):
        """batchify data for conversation after process.

        Args:
            batch (list): processed batch dataset.

        Returns:
            batch data for the system to train conversation part.
        """
        raise NotImplementedError('supervised must implement conv_batchify() method')

    def rec_process_fn(self):
        """Process whole data for recommendation before batch_fn.

        Returns:
            processed dataset. Defaults to return the same as `self.dataset`.

        """
        return self.dataset

    def rec_batchify(self, batch):
        """batchify data for recommendation after process.

        Args:
            batch (list): processed batch dataset.

        Returns:
            batch data for the system to train recommendation part.
        """
        raise NotImplementedError('supervised must implement rec_batchify() method')

    def policy_process_fn(self):
        """Process whole data for policy before batch_fn.

        Returns:
            processed dataset. Defaults to return the same as `self.dataset`.

        """
        return self.dataset

    def policy_batchify(self, batch):
        """batchify data for policy after process.

        Args:
            batch (list): processed batch dataset.

        Returns:
            batch data for the system to train policy part.
        """
        raise NotImplementedError('supervised must implement policy_batchify() method')

    def retain_recommender_target(self):
        """keep data whose role is recommender.

        Returns:
            Recommender part of ``self.dataset``.

        """
        dataset = []
        for conv_dict in tqdm(self.dataset):
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
