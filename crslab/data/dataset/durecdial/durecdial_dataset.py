# @Time   : 2020/12/21
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/12/21, 2020/12/22
# @Author : Kun Zhou, Xiaolei Wang
# @Email  : francis_kun_zhou@163.com, wxl1999@foxmail.com

import json
import os
from copy import copy

from loguru import logger
from tqdm import tqdm

from crslab.config import DATASET_PATH
from crslab.data.dataset.base_dataset import BaseDataset
from .resource import resources


class DuRecDialDataset(BaseDataset):
    def __init__(self, opt, tokenize, restore=False, save=False):
        resource = resources[tokenize]
        self.special_token_idx = resource['special_token_idx']
        self.unk_token_idx = self.special_token_idx['unk']
        dpath = os.path.join(DATASET_PATH, 'durecdial', tokenize)
        super().__init__(opt, dpath, resource, restore, save)

    def _load_data(self):
        train_data, valid_data, test_data = self._load_raw_data()
        self._load_vocab()
        self._load_other_data()

        vocab = {
            'tok2ind': self.tok2ind,
            'ind2tok': self.ind2tok,
            'vocab_size': len(self.tok2ind),
            'n_entity': self.n_entity,
            'n_word': self.n_word,
            'id2entity': self.id2entity
        }
        vocab.update(self.special_token_idx)

        return train_data, valid_data, test_data, vocab

    def _load_raw_data(self):
        with open(os.path.join(self.dpath, 'train_data.json'), 'r', encoding='utf-8') as f:
            train_data = json.load(f)
            logger.debug(f"[Load train data from {os.path.join(self.dpath, 'train_data.json')}]")
        with open(os.path.join(self.dpath, 'valid_data.json'), 'r', encoding='utf-8') as f:
            valid_data = json.load(f)
            logger.debug(f"[Load valid data from {os.path.join(self.dpath, 'valid_data.json')}]")
        with open(os.path.join(self.dpath, 'test_data.json'), 'r', encoding='utf-8') as f:
            test_data = json.load(f)
            logger.debug(f"[Load test data from {os.path.join(self.dpath, 'test_data.json')}]")

        return train_data, valid_data, test_data

    def _load_vocab(self):
        self.tok2ind = json.load(open(os.path.join(self.dpath, 'token2id.json'), 'r', encoding='utf-8'))
        self.ind2tok = {idx: word for word, idx in self.tok2ind.items()}

        logger.debug(f"[Load vocab from {os.path.join(self.dpath, 'token2id.json')}]")
        logger.debug(f"[The size of token2index dictionary is {len(self.tok2ind)}]")
        logger.debug(f"[The size of index2token dictionary is {len(self.ind2tok)}]")

    def _load_other_data(self):
        # entity kg
        with open(os.path.join(self.dpath, 'entity2id.json'), encoding='utf-8') as f:
            self.entity2id = json.load(f)  # {entity: entity_id}
        self.id2entity = {idx: entity for entity, idx in self.entity2id.items()}
        self.n_entity = max(self.entity2id.values()) + 1
        # {head_entity_id: [(relation_id, tail_entity_id)]}
        self.entity_kg = open(os.path.join(self.dpath, 'entity_subkg.txt'), encoding='utf-8')
        logger.debug(
            f"[Load entity dictionary and KG from {os.path.join(self.dpath, 'entity2id.json')} and {os.path.join(self.dpath, 'entity_subkg.txt')}]")

        # hownet
        # {concept: concept_id}
        with open(os.path.join(self.dpath, 'word2id.json'), 'r', encoding='utf-8') as f:
            self.word2id = json.load(f)
        self.n_word = max(self.word2id.values()) + 1
        # {concept \t relation\t concept}
        self.word_kg = open(os.path.join(self.dpath, 'hownet_subkg.txt'), encoding='utf-8')
        logger.debug(
            f"[Load word dictionary and KG from {os.path.join(self.dpath, 'word2id.json')} and {os.path.join(self.dpath, 'hownet_subkg.txt')}]")

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
        """process raw data

        Args:
            raw_data (list of dict): {
                'conv_id' (int):
                'messages' (list of dict): {
                    'local_id' (int): id of current utterance
                    'role' (str): 'Seeker' or 'Recommender'
                    'text' (list of str): utterance which has benn tokenized into tokens
                    'movie' (list of str): mentioned movies in text
                    'entity' (list of str): mentioned entities of dbpedia in text
                    'word' (list of str): mentioned words of conceptnet in text
                }
            }

        Returns:
            list of dict: {
                'role' (str): 'Seeker' or 'Recommender';
                'context_tokens' (list of list int): token ids of the preprocessed contextual dialog;
                'response' (list of int): token ids of the ground-truth response;
                'items' (list of int): item ids mentioned in current turn, we only keep those in dbpedia for comparison;
                'context_entities' (list of int): if necessary, id of entities in context;
                'context_words' (list of int): if necessary, id of words in context;
            }
        """
        augmented_convs = [self._convert_to_id(conversation) for conversation in tqdm(raw_data)]
        augmented_conv_dicts = []
        for conv in tqdm(augmented_convs):
            augmented_conv_dicts.extend(self._augment_and_add(conv))
        return augmented_conv_dicts

    def _convert_to_id(self, conversation):
        """
        convert token/word/entity/movie into ids;
        """
        augmented_convs = []
        last_role = None
        for utt in conversation['dialog']:
            assert utt['role'] != last_role, print(utt)

            text_token_ids = [self.tok2ind.get(word, self.unk_token_idx) for word in utt["text"]]
            item_ids = [self.entity2id[movie] for movie in utt['item'] if movie in self.entity2id]
            entity_ids = [self.entity2id[entity] for entity in utt['entity'] if entity in self.entity2id]
            word_ids = [self.word2id[word] for word in utt['word'] if word in self.word2id]

            augmented_convs.append({
                "role": utt["role"],
                "text": text_token_ids,
                "entity": entity_ids,
                "movie": item_ids,
                "word": word_ids
            })
            last_role = utt["role"]

        return augmented_convs

    def _augment_and_add(self, raw_conv_dict):
        """
        augment one conversation into several instances;
        """
        augmented_conv_dicts = []
        context_tokens, context_entities, context_words, context_items = [], [], [], []
        entity_set, word_set = set(), set()
        for i, conv in enumerate(raw_conv_dict):
            text_tokens, entities, movies, words = conv["text"], conv["entity"], conv["movie"], conv["word"]
            if len(context_tokens) > 0:
                conv_dict = {
                    'role': conv['role'],
                    "context_tokens": copy(context_tokens),
                    "response": text_tokens,
                    "context_entities": copy(context_entities),
                    "context_words": copy(context_words),
                    'context_items': copy(context_items),
                    "items": movies
                }
                augmented_conv_dicts.append(conv_dict)

            context_tokens.append(text_tokens)
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
        """process side data

        Returns:
            dict: {
                'entity_kg' (list of tuple): entity knowledge graph;
                'word_kg' (list of tuple): word knowledge graph;
                'item_entity_ids' (list of int): entity id of each item
            }
        """
        processed_entity_kg = self._entity_kg_process()
        logger.debug("[Finish entity KG process]")
        processed_word_kg = self._word_kg_process()
        logger.debug("[Finish word KG process]")
        with open(os.path.join(self.dpath, 'item_ids.json'), 'r', encoding='utf-8') as f:
            item_entity_ids = json.load(f)
        logger.debug('[Load movie entity ids]')

        side_data = {
            "entity_kg": processed_entity_kg,
            "word_kg": processed_word_kg,
            "item_entity_ids": item_entity_ids,
        }
        return side_data

    def _entity_kg_process(self):
        """get entity kg edge information

        Args:

        Returns:
            list: edge list [(head_entity_id, tail_entity_id, new_relation_id)]
        """
        edge_list = []  # [(entity, entity, relation)]
        for line in self.entity_kg:
            triple = line.strip().split('\t')
            e0 = self.entity2id[triple[0]]
            e1 = self.entity2id[triple[2]]
            r = triple[1]
            edge_list.append((e0, e1, r))
            edge_list.append((e1, e0, r))
            edge_list.append((e0, e0, 'SELF_LOOP'))
            if e1 != e0:
                edge_list.append((e1, e1, 'SELF_LOOP'))

        relation2id, edges = dict(), set()
        for h, t, r in edge_list:
            if r not in relation2id:
                relation2id[r] = len(relation2id)
            edges.add((h, t, relation2id[r]))

        return {
            'edge': list(edges),
            'n_relation': len(relation2id)
        }

    def _word_kg_process(self):
        """return [(head_word, tail_word)]"""
        edges = set()  # {(entity, entity)}
        for line in self.word_kg:
            triple = line.strip().split('\t')
            e0 = self.word2id[triple[0]]
            e1 = self.word2id[triple[2]]
            edges.add((e0, e1))
            edges.add((e1, e0))
        # edge_set = [[co[0] for co in list(edges)], [co[1] for co in list(edges)]]
        return list(edges)
