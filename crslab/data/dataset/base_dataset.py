# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/23, 2020/11/26
# @Author : Kun Zhou, Xiaolei Wang
# @Email  : francis_kun_zhou@163.com, wxl1999@foxmail.com

import os
import pickle as pkl
from abc import ABC, abstractmethod

from loguru import logger


class BaseDataset(ABC):
    """:class:`Dataset` stores the original dataset in memory.
    It provides many useful functions for data preprocessing. Finally, the dataset are preprocessed as
    {
        'dialog_context': the preprocessed contextual dialog;
        'interaction_context': if necessary, the preprocessed interaction history;
        'entity_context': if necessary, the entities in context;
        'word_context': if necessary, the words in context;
        'rec_item': the recommended item in this turn;
        'response': the ground-truth response;
        'guiding_thread': the guiding topic;
        'entity_knowledge_graph': if necessary, entity knowledge graph as side information;
        'word_knowledge_graph': if necessary, word knowledge graph as side information;
    }

    Args:
        config (Config): Global configuration object.

    Attributes:
    """

    def __init__(self, config, restore=False, save=False):
        self.config = config
        self.dataset_name = config['dataset']
        self.data_path = config['data_path']
        self.restore = restore
        self.save = save
        self.train_data, self.dev_data, self.test_data, self.side_data = self._from_scratch()

    def _from_scratch(self):
        """Load dataset from scratch.
        Initialize attributes firstly, then load data from atomic files, pre-process the dataset lastly.
        """
        logger.debug('Loading {} from scratch', self.__class__)
        if not self.restore:
            self._load_data()
            train_data, dev_data, test_data, side_data = self._data_preprocess()
        else:
            train_data, dev_data, test_data, side_data = self._load_from_restore()

        if self.save:
            self._save_preprocessed_data((train_data, dev_data, test_data, side_data))

        return train_data, dev_data, test_data, side_data

    @abstractmethod
    def _load_data(self):
        raise NotImplementedError

    @abstractmethod
    def _data_preprocess(self):
        raise NotImplementedError

    def _load_from_restore(self, file_name="all_data.pkl"):
        """Restore saved dataset from ``saved_dataset``.

        Args:
            file_name (str): file for the saved dataset.
        """
        logger.debug('Restoring dataset from [{}]'.format(file_name))
        if not os.path.exists(os.path.join(self.data_path, file_name)):
            raise ValueError('filepath [{}] does not exist'.format(file_name))

        with open(os.path.join(self.data_path, file_name), 'rb') as file:
            train_data, dev_data, test_data, side_data = pkl.load(file)

        return train_data, dev_data, test_data, side_data

    def _save_preprocessed_data(self, data, file_name="all_data.pkl"):
        save_path = os.path.join(self.data_path, file_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        pkl.dump(data, open(save_path, 'wb'))

    @staticmethod
    def add_start_end_token_idx(vec: list, add_start=False, start_idx=None, add_end=False, end_idx=None):
        if add_start:
            vec.insert(0, start_idx)
        if add_end:
            vec.append(end_idx)
        return vec

    @staticmethod
    def truncate(vec, max_length, truncate_head=False):
        """Check that vector is truncated correctly."""
        if max_length is None:
            return vec
        if len(vec) <= max_length:
            return vec
        if truncate_head:
            return vec[-max_length:]
        else:
            return vec[:max_length]
