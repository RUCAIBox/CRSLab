# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/23, 2020/12/13
# @Author : Kun Zhou, Xiaolei Wang
# @Email  : francis_kun_zhou@163.com, wxl1999@foxmail.com

import os
import pickle as pkl
from abc import ABC, abstractmethod

import numpy as np
from loguru import logger

from crslab.download import build


class BaseDataset(ABC):
    def __init__(self, opt, dpath, resource, restore=False, save=False):
        self.opt = opt
        self.dpath = dpath

        # download
        dfile = resource['file']
        build(dpath, dfile, version=resource['version'])

        if not restore:
            # load and process
            train_data, valid_data, test_data, self.vocab = self._load_data()
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
            self.train_data, self.valid_data, self.test_data, self.side_data, self.vocab = self._load_from_restore()

        if save:
            data = (self.train_data, self.valid_data, self.test_data, self.side_data, self.vocab)
            self._save_to_one(data)

    @abstractmethod
    def _load_data(self):
        """return train, valid, test data and vocab(all kinds of size and idx and map between token and idx)"""
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
                    'role' (str): 'Seeker' or 'Recommender';
                    'user_profile' (list of list of int): id of tokens of sentences of user profile
                    'context_tokens' (list of list int): token ids of the preprocessed contextual dialog;
                    'response' (list of int): token ids of the ground-truth response;
                    'interaction_history' (list of int): id of items which have interaction of the user in current turn;
                    'context_items' (list of int): item ids mentioned in context
                    'items' (list of int): item ids mentioned in current turn, we only keep those in entity kg for comparison;
                    'context_entities' (list of int): if necessary, id of entities in context;
                    'context_words' (list of int): if necessary, id of words in context;
                    'context_policy' (list of list of list): policy of each context turn, ont turn may have several policies, where first is action and second is keyword;
                    'target' (list): policy of current turn;
                    'final' (list): final goal for current turn;
                }
                side_data: {
                    'entity_kg' (dict): {
                        'edge': edges of entity kg;
                        'n_relation': #relation;
                    } if necessary
                    'word_kg' (list of tuple): if necessary, edges of word knowledge graph;
                    'item_entity_ids' (list of int): if necessary, entity id of each item;
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
