# @Time   : 2021/8/2
# @Author : Chenzhan Shang
# @Email  : czshang@outlook.com

r"""
Last.FM
======
References:
    Last.FM Dataset:
    https://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip

"""

import os
import json
from loguru import logger

from crslab.config import DATASET_PATH
from crslab.dataset.base import AttributeBaseDataset


class LastFMDataset(AttributeBaseDataset):
    """

    Attributes:
        train_data: train dataset.
        valid_data: valid dataset.
        test_data: test dataset.

    """
    def __init__(self, opt, tokenize, restore=False, save=False):
        resource = None
        dpath = os.path.join(DATASET_PATH, "lastfm")
        super(LastFMDataset, self).__init__(opt, dpath, resource, restore, save)

    def _load_and_preprocess(self):
        entities, relations = self._load_kg_data()
        kg = self._process_kg(entities, relations)
        interactions = self._load_interaction_data()
        return kg, interactions

    def _load_kg_data(self):
        with open(os.path.join(self.dpath, 'user_map.json'), 'r', encoding='utf-8') as f:
            user_map = json.load(f)
        with open(os.path.join(self.dpath, 'item_map.json'), 'r', encoding='utf-8') as f:
            item_map = json.load(f)
        with open(os.path.join(self.dpath, 'merged_tag_map.json'), 'r', encoding='utf-8') as f:
            attribute_map = json.load(f)
        with open(os.path.join(self.dpath, 'user_dict.json'), 'r', encoding='utf-8') as f:
            user_dict = json.load(f)
        with open(os.path.join(self.dpath, 'item_dict.json'), 'r', encoding='utf-8') as f:
            item_dict = json.load(f)
        with open(os.path.join(self.dpath, 'user_item.json'), 'r', encoding='utf-8') as f:
            ui_dict = json.load(f)
        entities = {
            'user': list(user_map.values()),
            'item': list(item_map.values()),
            'attribute': list(attribute_map.values())
        }
        logger.debug(f"[Load entity data of knowledge graph]")

        relations = {
            'user2user': dict(),  # friend
            'user2item': dict(),  # interact
            'user2attribute': dict(),  # like
            'item2attribute': dict()  # belong_to
        }
        for user in user_dict.keys():
            relations['user2user'][int(user)] = user_dict[user]['friends']
            relations['user2attribute'][int(user)] = user_dict[user]['like']
        for item in item_dict.keys():
            relations['item2attribute'][int(item)] = item_dict[item]['feature_index']
        for user in ui_dict.keys():
            relations['user2item'][int(user)] = ui_dict[user]
        logger.debug(f"[Load relation data of knowledge graph]")
        return entities, relations

    def _process_kg(self, entities, relations):
        kg = self._get_template(entities['user'], entities['item'], entities['attribute'])
        edge_counter = 0
        for user, users in relations['user2user'].items():
            for friend in users:
                kg['user'][user]['friend'].append(friend)
                kg['user'][friend]['friend'].append(user)
                edge_counter += 2
        for user, items in relations['user2item'].items():
            for item in items:
                kg['user'][user]['interact'].append(item)
                kg['item'][item]['interact'].append(user)
                edge_counter += 2
        for user, attributes in relations['user2attribute'].items():
            for attribute in attributes:
                kg['user'][user]['like'].append(attribute)
                kg['attribute'][attribute]['like'].append(user)
                edge_counter += 2
        for item, attributes in relations['item2attribute'].items():
            for attribute in attributes:
                kg['item'][item]['belong_to'].append(attribute)
                kg['attribute'][attribute]['belong_to'].append(item)
                edge_counter += 2
        logger.debug(f"[Build knowledge graph with {edge_counter} edges]")

        # remove duplicates
        for entity_type in kg:
            for entity_id in kg[entity_type]:
                for relation in kg[entity_type][entity_id]:
                    data = kg[entity_type][entity_id][relation]
                    kg[entity_type][entity_id][relation] = list(sorted(set(data)))
        logger.debug(f"[Remove duplicates from knowledge graph]")
        return kg

    def _load_interaction_data(self):
        with open(os.path.join(self.dpath, 'review_dict_train.json'), 'r', encoding='utf-8') as f:
            train_raw = json.load(f)
        with open(os.path.join(self.dpath, 'review_dict_valid.json'), 'r', encoding='utf-8') as f:
            valid_raw = json.load(f)
        with open(os.path.join(self.dpath, 'review_dict_test.json'), 'r', encoding='utf-8') as f:
            test_raw = json.load(f)
        train, valid, test = dict(), dict(), dict()
        for user in train_raw:
            train[int(user)] = train_raw[user]
        for user in valid_raw:
            valid[int(user)] = valid_raw[user]
        for user in test_raw:
            test[int(user)] = test_raw[user]
        return {
            'train': train,
            'valid': valid,
            'test': test
        }

    def generate_data(self, mode):
        """Generate train and valid data for Bayesian Personalized Ranking objective.

        train data (list of tuple): [(
            user (int): user of the session.
            positive item (int): the ground-truth item of the session.
            negative item (int): the non-interacted item of the user.
            attribute-aware negative item (int): the non-interacted item of the user in the candidate item set.
            preference (list of int): partially known preferred attributes of the user.
        )]

        valid data (list of tuple): [(
            user (int): user of the session.
            positive item (int): the ground-truth item of the session.
            candidate item set (int): the candidate items satisfy the partially known preferred attributes.
            preference (list of int): partially known preferred attributes of the user.
        )]

        """
        assert mode in ('train', 'valid')
        # TODO - design train - merge valid
        pass
