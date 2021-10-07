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

from crslab.utils.download import build
from crslab.utils import DatasetType, LanguageType


class TextBaseDataset(ABC):
    """Abstract class of dataset

    Notes:
        ``'embedding'`` can be specified in config to use pretrained word embedding.

    """

    def __init__(self, opt, dpath, resource, restore=False, save=False):
        """Download resource, load, process data. Support restore and save processed dataset.

        Args:
            opt (Config or dict): config for dataset or the whole system.
            dpath (str): where to store dataset.
            resource (dict): version, download file and special token idx of tokenized dataset.
            restore (bool, optional): whether to restore saved dataset which has been processed. Defaults to False.
            save (bool, optional): whether to save dataset after processing. Defaults to False.

        """
        self.opt = opt
        self.dpath = dpath

        # download
        dfile = resource['file']
        build(dpath, dfile, version=resource['version'])

        if not restore:
            # load and process
            train_data, valid_data, test_data, vocab = self._load_data()
            logger.info('[Finish data load]')
            self.train_data, self.valid_data, self.test_data, self.other_data = self._data_preprocess(train_data,
                                                                                                      valid_data,
                                                                                                      test_data)
            embedding = opt.get('embedding', None)
            if embedding:
                self.other_data["embedding"] = np.load(os.path.join(self.dpath, embedding))
                logger.debug(f'[Load pretrained embedding {embedding}]')

            self.other_data['vocab'] = vocab
            logger.info('[Finish data preprocess]')
        else:
            self.train_data, self.valid_data, self.test_data, self.other_data = self._restore_data()

        if save:
            data = (self.train_data, self.valid_data, self.test_data, self.other_data)
            self._save_data(data)

        self._dataset_type = self._set_dataset_type()
        assert isinstance(self._dataset_type, DatasetType)
        self._language_type = self._set_language_type()
        assert isinstance(self._language_type, LanguageType)

    @property
    def dataset_type(self):
        return self._dataset_type

    @abstractmethod
    def _set_dataset_type(self) -> DatasetType:
        pass

    @property
    def language_type(self):
        return self._language_type

    @abstractmethod
    def _set_language_type(self) -> LanguageType:
        pass

    @abstractmethod
    def _load_data(self):
        """Load dataset.

        Returns:
            (any, any, any, dict):

            raw train, valid and test data.

            vocab: all kinds of useful size, idx and map between token and idx.

        """
        pass

    @abstractmethod
    def _data_preprocess(self, train_data, valid_data, test_data):
        """Process raw train, valid, test data.

        Args:
            train_data: train dataset.
            valid_data: valid dataset.
            test_data: test dataset.

        Returns:
            (list of dict, dict):

            train/valid/test_data, each dict is in the following format::

                 {
                    'role' (str):
                        'Seeker' or 'Recommender',
                    'user_profile' (list of list of int):
                        id of tokens of sentences of user profile,
                    'context_tokens' (list of list int):
                        token ids of preprocessed contextual dialogs,
                    'response' (list of int):
                        token ids of the ground-truth response,
                    'interaction_history' (list of int):
                        id of items which have interaction of the user in current turn,
                    'context_items' (list of int):
                        item ids mentioned in context,
                    'items' (list of int):
                        item ids mentioned in current turn, we only keep
                        those in entity kg for comparison,
                    'context_entities' (list of int):
                        if necessary, id of entities in context,
                    'context_words' (list of int):
                        if necessary, id of words in context,
                    'context_policy' (list of list of list):
                        policy of each context turn, one turn may have several policies,
                        where first is action and second is keyword,
                    'target' (list): policy of current turn,
                    'final' (list): final goal for current turn
                }

            other_data, which is in the following format::

                {
                    'entity_kg': {
                        'edge' (list of tuple): (head_entity_id, tail_entity_id, relation_id),
                        'n_relation' (int): number of distinct relations,
                        'entity' (list of str): str of entities, used for entity linking
                    },
                    'word_kg': {
                        'edge' (list of tuple): (head_entity_id, tail_entity_id),
                        'entity' (list of str): str of entities, used for entity linking
                    }
                    'item_entity_ids' (list of int): entity id of each item;
                }

        """
        pass

    def _restore_data(self, file_name="all_data.pkl"):
        """Restore saved dataset.

        Args:
            file_name (str): file of saved dataset. Defaults to "all_data.pkl".

        """
        if not os.path.exists(os.path.join(self.dpath, file_name)):
            raise ValueError(f'Saved dataset [{file_name}] does not exist')
        with open(os.path.join(self.dpath, file_name), 'rb') as f:
            dataset = pkl.load(f)
        logger.info(f'Restore dataset from [{file_name}]')
        return dataset

    def _save_data(self, data, file_name="all_data.pkl"):
        """Save all processed dataset and vocab into one file.

        Args:
            data (tuple): all dataset and vocab.
            file_name (str, optional): file to save dataset. Defaults to "all_data.pkl".

        """
        if not os.path.exists(self.dpath):
            os.makedirs(self.dpath)
        save_path = os.path.join(self.dpath, file_name)
        with open(save_path, 'wb') as f:
            pkl.dump(data, f)
        logger.info(f'[Save dataset to {file_name}]')


class AttributeBaseDataset(ABC):
    """Abstract class of dataset

    """

    def __init__(self, opt, dpath, resource, restore=False, save=False):
        """Download resource, load, process data. Support restore and save processed dataset.

        Args:
            opt (Config or dict): config for dataset or the whole system.
            dpath (str): where to store dataset.
            resource (dict): version, download file and special token idx of tokenized dataset.
            restore (bool): whether to restore saved dataset which has been processed. Defaults to False.
            save (bool): whether to save dataset after processing. Defaults to False.

        """
        self.opt = opt
        self.dpath = dpath

        # download - TODO
        # dfile = resource['file']
        # build(dpath, dfile, version=resource['version'])

        if not restore:
            # load and process
            self.kg, self.interactions = self._load_and_preprocess()
            logger.info('[Finish data preprocess]')
        else:
            self.kg, self.interactions = self._restore_data()

        if save:
            data = (self.kg, self.interactions)
            self._save_data(data)
        self._dataset_type = self._set_dataset_type()
        assert isinstance(self._dataset_type, DatasetType)
        self._language_type = self._set_language_type()
        assert isinstance(self._language_type, LanguageType)

    @property
    def dataset_type(self):
        return self._dataset_type

    @abstractmethod
    def _set_dataset_type(self) -> DatasetType:
        pass

    @property
    def language_type(self):
        return self._language_type

    @abstractmethod
    def _set_language_type(self) -> LanguageType:
        pass

    @abstractmethod
    def _load_and_preprocess(self):
        """Load and preprocess dataset.

        Returns:
            (dict, dict):

            knowledge graph, which is in the following format::

            {
                'user': {
                    user_id: {
                        'interact' (list of int): ids of items which the user interacts with.
                        'friend' (list of int): ids of users which are friends of the user.
                        'like' (list of int): ids of attributes which the user likes.
                    }
                },
                'item': {
                    item_id: {
                        'belong_to' (list of int): ids of attributes which belong to the item.
                        'interact' (list of int): ids of users which the item interacts with.
                    }
                },
                'attribute': {
                    attribute_id: {
                        'like' (list of int): ids of users which like the attribute.
                        'belong_to' (list of int): ids of items which the attribute belongs to.
                    }
                }
            }

            interaction records, which is in the following format::

            {
                'train': {
                    user_id (list of int): ids of items which the user interacts with.
                },
                'valid': {
                    user_id (list of int): ids of items which the user interacts with.
                },
                'test': {
                    user_id (list of int): ids of items which the user interacts with.
                }
            }

        """
        pass

    def _get_template(self, users, items, attributes):
        template = {
            'user': dict(),
            'item': dict(),
            'attribute': dict()
        }
        for user_id in users:
            template['user'][user_id] = {
                'interact': [],
                'friend': [],
                'like': []
            }
        for item_id in items:
            template['item'][item_id] = {
                'belong_to': [],
                'interact': []
            }
        for attribute_id in attributes:
            template['attribute'][attribute_id] = {
                'like': [],
                'belong_to': []
            }
        return template

    def _restore_data(self, file_name="all_data.pkl"):
        """Restore saved dataset.

        Args:
            file_name (str): file of saved dataset. Defaults to "all_data.pkl".

        """
        if not os.path.exists(os.path.join(self.dpath, file_name)):
            raise ValueError(f'Saved dataset [{file_name}] does not exist')
        with open(os.path.join(self.dpath, file_name), 'rb') as f:
            dataset = pkl.load(f)
        logger.info(f'Restore dataset from [{file_name}]')
        return dataset

    def _save_data(self, data, file_name="all_data.pkl"):
        """Save all processed dataset and vocab into one file.

        Args:
            data (tuple): all dataset and vocab.
            file_name (str, optional): file to save dataset. Defaults to "all_data.pkl".

        """
        if not os.path.exists(self.dpath):
            os.makedirs(self.dpath)
        save_path = os.path.join(self.dpath, file_name)
        with open(save_path, 'wb') as f:
            pkl.dump(data, f)
        logger.info(f'[Save dataset to {file_name}]')
