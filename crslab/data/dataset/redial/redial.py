# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/23, 2021/1/3, 2020/12/19
# @Author : Kun Zhou, Xiaolei Wang, Yuanhang Zhou
# @Email  : francis_kun_zhou@163.com, wxl1999@foxmail.com, sdzyh002@gmail

# UPDATE:
# @Time   : 2023/6/14
# @Author : Xinyu Tang
# @Email  : txy20010310@163.com

r"""
ReDial
======
References:
    Li, Raymond, et al. `"Towards deep conversational recommendations."`_ in NeurIPS 2018.

.. _`"Towards deep conversational recommendations."`:
   https://papers.nips.cc/paper/2018/hash/800de15c79c8d840f4e78d3af937d4d4-Abstract.html

"""

import json
import re
import os
from collections import defaultdict
from copy import copy

from loguru import logger
from tqdm import tqdm

from crslab.config import DATASET_PATH
from crslab.data.dataset.base import BaseDataset
from .resources import resources


class ReDialDataset(BaseDataset):
    """

    Attributes:
        train_data: train dataset.
        valid_data: valid dataset.
        test_data: test dataset.
        vocab (dict): ::

            {
                'tok2ind': map from token to index,
                'ind2tok': map from index to token,
                'entity2id': map from entity to index,
                'id2entity': map from index to entity,
                'word2id': map from word to index,
                'vocab_size': len(self.tok2ind),
                'n_entity': max(self.entity2id.values()) + 1,
                'n_word': max(self.word2id.values()) + 1,
            }

    Notes:
        ``'unk'`` must be specified in ``'special_token_idx'`` in ``resources.py``.

    """

    def __init__(self, opt, tokenize, restore=False, save=False):
        """Specify tokenized resource and init base dataset.

        Args:
            opt (Config or dict): config for dataset or the whole system.
            tokenize (str): how to tokenize dataset.
            restore (bool): whether to restore saved dataset which has been processed. Defaults to False.
            save (bool): whether to save dataset after processing. Defaults to False.

        """
        resource = resources[tokenize]
        self.special_token_idx = resource['special_token_idx']
        self.unk_token_idx = self.special_token_idx['unk']
        dpath = os.path.join(DATASET_PATH, opt['dataset'])
        super().__init__(opt, dpath, resource, restore, save)

    def _load_data(self):
        train_data, valid_data, test_data = self._load_raw_data()
        # self._load_vocab()
        self._load_other_data()

        vocab = {
            # 'tok2ind': self.tok2ind,
            # 'ind2tok': self.ind2tok,
            'entity2id': self.entity2id,
            'id2entity': self.id2entity,
            'word2id': self.word2id,
            # 'vocab_size': len(self.tok2ind),
            'n_entity': self.n_entity,
            'n_word': self.n_word,
        }
        vocab.update(self.special_token_idx)

        return train_data, valid_data, test_data, vocab

    def _load_raw_data(self):
        # load train/valid/test data
        train_data = []
        valid_data = []
        test_data = []
        
        with open(os.path.join(self.dpath, 'train_data.jsonl'), 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                data = json.loads(line)
                train_data.append(data)
            logger.debug(f"[Load train data from {os.path.join(self.dpath, 'train_data.json')}]")
        with open(os.path.join(self.dpath, 'valid_data.jsonl'), 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                data = json.loads(line)
                valid_data.append(data)
            valid_data.append(data)
            logger.debug(f"[Load valid data from {os.path.join(self.dpath, 'valid_data.json')}]")
        with open(os.path.join(self.dpath, 'test_data.jsonl'), 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                data = json.loads(line)
                test_data.append(data)
            logger.debug(f"[Load test data from {os.path.join(self.dpath, 'test_data.json')}]")

        return train_data, valid_data, test_data

    def _load_vocab(self):
        self.tok2ind = json.load(open(os.path.join(self.dpath, 'token2id.json'), 'r', encoding='utf-8'))
        self.ind2tok = {idx: word for word, idx in self.tok2ind.items()}

        logger.debug(f"[Load vocab from {os.path.join(self.dpath, 'token2id.json')}]")
        logger.debug(f"[The size of token2index dictionary is {len(self.tok2ind)}]")
        logger.debug(f"[The size of index2token dictionary is {len(self.ind2tok)}]")

    def _load_other_data(self):
        # dbpedia
        self.entity2id = json.load(
            open(os.path.join(self.dpath, 'entity2id.json'), 'r', encoding='utf-8'))  # {entity: entity_id}
        self.id2entity = {idx: entity for entity, idx in self.entity2id.items()}
        self.n_entity = max(self.entity2id.values()) + 1
        # {head_entity_id: [(relation_id, tail_entity_id)]}
        self.entity_kg = json.load(open(os.path.join(self.dpath, 'kg.json'), 'r', encoding='utf-8'))
        logger.debug(
            f"[Load entity dictionary and KG from {os.path.join(self.dpath, 'entity2id.json')} and {os.path.join(self.dpath, 'dbpedia_subkg.json')}]")

        # conceptNet
        # {concept: concept_id}
        self.word2id = json.load(open(os.path.join(self.dpath, 'concept2id.json'), 'r', encoding='utf-8'))
        self.n_word = max(self.word2id.values()) + 1
        # {relation\t concept \t concept}
        self.word_kg = open(os.path.join(self.dpath, 'conceptnet_subkg.txt'), 'r', encoding='utf-8')
        logger.debug(
            f"[Load word dictionary and KG from {os.path.join(self.dpath, 'concept2id.json')} and {os.path.join(self.dpath, 'conceptnet_subkg.txt')}]")
        with open(os.path.join(self.dpath, 'id2info.json'), 'r', encoding='utf-8') as f:
            id2info = json.load(f)
        self.id2name = {}
        for id, info in id2info.items():
            self.id2name[id] = info['name']


    def _data_preprocess(self, train_data, valid_data, test_data):
        processed_train_data = self._raw_data_process(train_data)
        logger.debug("[Finish train data process]")
        processed_valid_data = self._raw_data_process(valid_data)
        logger.debug("[Finish valid data process]")
        processed_test_data = self._raw_data_process(test_data)
        logger.debug("[Finish test data process]")
        processed_side_data = self._side_data_process()
        logger.debug("[Finish side data process]")
        return processed_train_data, processed_valid_data, processed_test_data, processed_side_data

    def _raw_data_process(self, raw_data):
        augmented_convs = [self._merge_conv_data(conversation) for conversation in tqdm(raw_data)]
        augmented_conv_dicts = []
        for conv in tqdm(augmented_convs):
            augmented_conv_dicts.extend(self._augment_and_add(conv))
        return augmented_conv_dicts

    def _merge_conv_data(self, dialog):
        movie_pattern = re.compile(r'@\d+')
        augmented_convs = []
        last_role = None
        user_id = dialog['initiatorWorkerId']
        recommender_id = dialog['respondentWorkerId']
        conversation_id = dialog['conversationId']
        
        for utt in dialog["messages"]:
            # text_token_ids = [self.tok2ind.get(word, self.unk_token_idx) for word in utt["text"]]
            turn_id = utt['turn_id']
            dialog_turn_id = str(conversation_id) + '_' + str(turn_id)
            
            text = utt['text']
            for pattern in re.findall(movie_pattern, text):
                if pattern.strip('@') in self.id2name:
                    text = text.replace(pattern, self.id2name[pattern.strip('@')])
            movie_ids = [self.entity2id[movie] for movie in utt['item'] if movie in self.entity2id]
            entity_ids = [self.entity2id[entity] for entity in utt['entity'] if entity in self.entity2id]
            word_ids = [self.word2id[word] for word in utt['word'] if word in self.word2id]
            
            role_id = utt['senderWorkerId']
            if role_id == recommender_id:
                role = 'Recommender'
            elif role_id == user_id:
                role = 'User'

            if role == last_role:
                augmented_convs[-1]["text"] += text
                augmented_convs[-1]["item"] += movie_ids
                augmented_convs[-1]["entity"] += entity_ids
                augmented_convs[-1]["word"] += word_ids
            else:
                augmented_convs.append({
                    "dialog_id": dialog_turn_id,
                    "role": role,
                    "text": text,
                    "entity": entity_ids,
                    "item": movie_ids,
                    "word": word_ids
                })
            last_role = role

        return augmented_convs

    def _augment_and_add(self, raw_conv_dict):
        augmented_conv_dicts = []
        context, context_entities, context_words, context_items = [], [], [], []
        entity_set, word_set = set(), set()
        for i, conv in enumerate(raw_conv_dict):
            text, entities, movies, words = conv["text"], conv["entity"], conv["item"], conv["word"]
            if len(context) > 0:
                conv_dict = {
                    "dialog_id": conv['dialog_id'],
                    "role": conv['role'],
                    "context": copy(context),
                    "response": text,
                    "context_entities": copy(context_entities),
                    "context_words": copy(context_words),
                    "context_items": copy(context_items),
                    "items": movies,
                }
                augmented_conv_dicts.append(conv_dict)

            context.append(text)
            context_items += movies
            for entity in entities + movies:
                if entity not in entity_set:
                    entity_set.add(entity)
                    context_entities.append(entity)
            for word in words:
                if word not in word_set:
                    word_set.add(word)
                    context_words.append(word)

        return augmented_conv_dicts

    def _side_data_process(self):
        processed_entity_kg = self._entity_kg_process()
        logger.debug("[Finish entity KG process]")
        processed_word_kg = self._word_kg_process()
        logger.debug("[Finish word KG process]")
        movie_entity_ids = json.load(open(os.path.join(self.dpath, 'item_ids.json'), 'r', encoding='utf-8'))
        logger.debug('[Load movie entity ids]')

        side_data = {
            "entity_kg": processed_entity_kg,
            "word_kg": processed_word_kg,
            "item_entity_ids": movie_entity_ids,
        }
        return side_data

    def _entity_kg_process(self, SELF_LOOP_ID=185):
        edge_list = []  # [(entity, entity, relation)]
        for entity in range(self.n_entity):
            if str(entity) not in self.entity_kg:
                continue
            edge_list.append((entity, entity, SELF_LOOP_ID))  # add self loop
            for tail_and_relation in self.entity_kg[str(entity)]:
                if entity != tail_and_relation[1] and tail_and_relation[0] != SELF_LOOP_ID:
                    edge_list.append((entity, tail_and_relation[1], tail_and_relation[0]))
                    edge_list.append((tail_and_relation[1], entity, tail_and_relation[0]))

        relation_cnt, relation2id, edges, entities = defaultdict(int), dict(), set(), set()
        for h, t, r in edge_list:
            relation_cnt[r] += 1
        for h, t, r in edge_list:
            if relation_cnt[r] > 1000:
                if r not in relation2id:
                    relation2id[r] = len(relation2id)
                edges.add((h, t, relation2id[r]))
                entities.add(self.id2entity[h])
                entities.add(self.id2entity[t])
        return {
            'edge': list(edges),
            'n_relation': len(relation2id),
            'entity': list(entities)
        }

    def _word_kg_process(self):
        edges = set()  # {(entity, entity)}
        entities = set()
        for line in self.word_kg:
            kg = line.strip().split('\t')
            entities.add(kg[1].split('/')[0])
            entities.add(kg[2].split('/')[0])
            e0 = self.word2id[kg[1].split('/')[0]]
            e1 = self.word2id[kg[2].split('/')[0]]
            edges.add((e0, e1))
            edges.add((e1, e0))
        # edge_set = [[co[0] for co in list(edges)], [co[1] for co in list(edges)]]
        return {
            'edge': list(edges),
            'entity': list(entities)
        }
    
    def get_attr_list(self):
        attr_list = ['genre', 'star', 'director']
        return attr_list
    
    def get_ask_instruction(self):
        ask_instruction = '''To recommend me items that I will accept, you can choose one of the following options.
A: ask my preference for genre
B: ask my preference for actor
C: ask my preference for director
D: I can directly give recommendations
Please enter the option character. Please only response a character.'''
        option2attr = {
            'A': 'genre',
            'B': 'star',
            'C': 'director',
            'D': 'recommend'
        }
        option2template = {
            'A': 'Which genre do you like?',
            'B': 'Which star do you like?',
            'C': 'Which director do you like?',
        }
        rec_instruction = 'Please give me 10 recommendations according to my preference (Format: no. title. No other things except the item list in your response). You can recommend mentioned items in our dialog.'
        
        ask_instruction_dict = {
            'ask_instruction': ask_instruction,
            'option2attr': option2attr,
            'option2template': option2template,
            'rec_instruction': rec_instruction
        }
        
        return ask_instruction_dict