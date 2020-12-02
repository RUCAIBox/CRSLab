# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/23, 2020/11/29
# @Author : Kun Zhou, Xiaolei Wang
# @Email  : francis_kun_zhou@163.com, wxl1999@foxmail.com

import os
import pickle as pkl
from abc import ABC, abstractmethod

import numpy as np
from loguru import logger


def add_start_end_token_idx(vec: list, add_start=False, start_token_idx=None, add_end=False, end_token_idx=None):
    if add_start:
        vec.insert(0, start_token_idx)
    if add_end:
        vec.append(end_token_idx)
    return vec


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
        opt (Config): Global configuration object.

    Attributes:
    """

    def __init__(self, opt, dpath, restore=False, save=False):
        self.opt = opt
        self.dpath = dpath

        if not restore:
            train_data, valid_data, test_data, self.tok2ind, self.ind2tok = self._load_data()
            logger.info('[Finish data load]')
            self.train_data, self.valid_data, self.test_data, self.side_data = self._data_preprocess(train_data,
                                                                                                     valid_data,
                                                                                                     test_data)
            embedding = opt.get('embedding', None)
            if embedding:
                self.side_data["embedding"] = np.load(os.path.join(self.dpath, embedding))
                logger.debug(f'[Load pretrained embedding {embedding}]')
            logger.info('[Finish data preprocess]')
        else:
            self.train_data, self.valid_data, self.test_data, self.side_data, self.tok2ind, self.ind2tok = self._load_from_restore()

        if save:
            data = (self.train_data, self.valid_data, self.test_data, self.side_data, self.tok2ind, self.ind2tok)
            self._save_to_one(data)

    @abstractmethod
    def _load_data(self):
        """return train, valid, test data and tok2ind, ind2tok"""
        pass

    @abstractmethod
    def _data_preprocess(self, train_data, valid_data, test_data):
        """return train, valid, test, side data"""
        pass

    def _load_from_restore(self, file_name="all_data.pkl"):
        """Restore saved dataset from ``saved_dataset``.

        Args:
            file_name (str): file for the saved dataset.
        """
        logger.info(f'Restore dataset from [{file_name}]')
        if not os.path.exists(os.path.join(self.dpath, file_name)):
            raise ValueError(f'Filepath [{file_name}] does not exist')
        with open(os.path.join(self.dpath, file_name), 'rb') as f:
            return pkl.load(f)

    def _save_to_one(self, data, file_name="all_data.pkl"):
        save_path = os.path.join(self.dpath, file_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(save_path, 'wb') as f:
            pkl.dump(data, f)
        logger.info(f'[Save dataset to {file_name}]')
