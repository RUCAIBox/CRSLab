# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/23, 2020/12/2
# @Author : Kun Zhou, Xiaolei Wang
# @Email  : francis_kun_zhou@163.com, wxl1999@foxmail.com

import os
import pickle as pkl
from abc import ABC, abstractmethod
from copy import copy

import numpy as np
from loguru import logger


def add_start_end_token_idx(vec: list, add_start=False, start_token_idx=None, add_end=False, end_token_idx=None):
    res = copy(vec)
    if add_start:
        res.insert(0, start_token_idx)
    if add_end:
        res.append(end_token_idx)
    return res


class BaseDataset(ABC):
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
        """preprocess train, valid, test data after load

        Args:
            train_data (list of dict):
            valid_data (list of dict):
            test_data (list of dict):

        Returns:
            list of dict: 
                train/valid/test_data: {
                    'context_tokens' (list of list int): the preprocessed contextual dialog;
                    'response' (list of int): the ground-truth response;
                    'items' (list of int): items to recommend in current turn;
                    'context_entities' (list of int): if necessary, the entities in context;
                    'context_words' (list of int): if necessary, the words in context;
                    'context_interactions' (): if necessary, the preprocessed interaction history;
                }
                side_data: {
                    'entity_kg' (list of tuple): if necessary, entity knowledge graph;
                    'word_kg' (list of tuple): if necessary, word knowledge graph;
                    'item_entity_ids' (list of int): if necessary, entity id of each item
                }
        """
        pass

    def _load_from_restore(self, file_name="all_data.pkl"):
        """Restore saved dataset from ``saved_dataset``.

        Args:
            file_name (str): file for the saved dataset.
        """
        if not os.path.exists(os.path.join(self.dpath, file_name)):
            raise ValueError(f'Saved dataset [{file_name}] does not exist')
        with open(os.path.join(self.dpath, file_name), 'rb') as f:
            dataset = pkl.load(f)
        logger.info(f'Restore dataset from [{file_name}]')
        return dataset

    def _save_to_one(self, data, file_name="all_data.pkl"):
        """save all data and vocab into one file

        Args:
            data (tuple): all data and vocab
            file_name (str, optional): Defaults to "all_data.pkl".
        """
        if not os.path.exists(self.dpath):
            os.makedirs(self.dpath)
        save_path = os.path.join(self.dpath, file_name)
        with open(save_path, 'wb') as f:
            pkl.dump(data, f)
        logger.info(f'[Save dataset to {file_name}]')
