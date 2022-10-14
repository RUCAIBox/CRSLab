# @Time   : 2020/12/19
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/12/20, 2021/1/2
# @Author : Kun Zhou, Xiaolei Wang
# @Email  : francis_kun_zhou@163.com, wxl1999@foxmail.com

# UPDATE
# @Time    :   2022/9/26
# @Author  :   Xinyu Tang
# @email   :   txy20010310@163.com

r"""
OpenDialKG
==========
References:
    Moon, Seungwhan, et al. `"Opendialkg: Explainable conversational reasoning with attention-based walks over knowledge graphs."`_ in ACL 2019.

.. _`"Opendialkg: Explainable conversational reasoning with attention-based walks over knowledge graphs."`:
   https://www.aclweb.org/anthology/P19-1081/

"""

import json
import os
from collections import defaultdict
from copy import copy
from http.client import NotConnected

import gensim
import numpy as np
from loguru import logger
from tqdm import tqdm

from crslab.config import DATASET_PATH, MODEL_PATH
from crslab.data.dataset.base import BaseDataset

from .resources import resources


class OpenDialKGDataset(BaseDataset):
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

    def __init__(self, opt, tokenize, crs_tokenizer, restore=False, save=False):
        """Specify tokenized resource and init base dataset.

        Args:
            opt (Config or dict): config for dataset or the whole system.
            tokenize (str): how to tokenize dataset.
            restore (bool): whether to restore saved dataset which has been processed. Defaults to False.
            save (bool): whether to save dataset after processing. Defaults to False.

        """
        if 'copy' in opt:
            self.copy = True
        else:
            self.copy = False

        if 'embedding' in opt:
            self.generate_embedding = True
        else:
            self.generate_embedding = False

        resource = resources['resource']
        self.special_token_idx = crs_tokenizer.special_token_idx
        self.unk_token_idx = self.special_token_idx['unk']
        self.tokenize = tokenize
        self.Tokenizer = crs_tokenizer
        dpath = os.path.join(DATASET_PATH, 'opendialkg')
        super().__init__(opt, dpath, resource, restore, save)

    def _load_data(self):
        train_data, valid_data, test_data, npy_dict = self._load_raw_data()
        self._load_vocab()
        self._load_other_data()

        vocab = {
            'tok2ind': self.tok2ind,
            'ind2tok': self.ind2tok,
            'entity2id': self.entity2id,
            'id2entity': self.id2entity,
            'word2id': self.word2id,
            'vocab_size': len(self.tok2ind),
            'n_entity': self.n_entity,
            'n_word': self.n_word,
            'word2vec': npy_dict['word2vec'],
            'copy_mask': npy_dict['copy_mask'],
            'special_token_idx': self.special_token_idx
        }

        return train_data, valid_data, test_data, vocab

    def _load_raw_data(self):
        # load train/valid/test data
        with open(os.path.join(self.dpath, 'train_data.json'), 'r', encoding='utf-8') as f:
            train_data = json.load(f)
            logger.debug(
                f"[Load train data from {os.path.join(self.dpath, 'train_data.json')}]")
        # split text
        processed_train_data = self.split_text(train_data)
        logger.info("[Finish train data split]")
        # generate tok2ind
        self.tok2ind = self.generate_tok2ind(processed_train_data)
        logger.info("[Finish generate train tok2ind]")
        # generate word2vec
        word_embedding = None
        if self.generate_embedding:
            word_embedding = self.generate_word2vec(processed_train_data)
            logger.info('[Finish generate word2vec]')
        # build copy_mask
        copy_mask = None
        if self.copy:
            copy_mask = self.generate_copy_mask(self.tok2ind, processed_train_data)
            logger.info('[Finish generate copy_mask]')

        with open(os.path.join(self.dpath, 'valid_data.json'), 'r', encoding='utf-8') as f:
            valid_data = json.load(f)
            logger.debug(
                f"[Load valid data from {os.path.join(self.dpath, 'valid_data.json')}]")
         # split_text
        processed_valid_data = self.split_text(valid_data)
        logger.info("[Finish valid data split]")

        with open(os.path.join(self.dpath, 'test_data.json'), 'r', encoding='utf-8') as f:
            test_data = json.load(f)
            logger.debug(
                f"[Load test data from {os.path.join(self.dpath, 'test_data.json')}]")
        # split_text
        processed_test_data = self.split_text(test_data)
        logger.info("[Finish test data split]")

        npy_dict = {'word2vec': word_embedding, 'copy_mask': copy_mask}

        return processed_train_data, processed_valid_data, processed_test_data, npy_dict

    def _load_vocab(self):
        self.ind2tok = {idx: word for word, idx in self.tok2ind.items()}

        logger.debug(
            f"[Load vocab from token2id]")
        logger.debug(
            f"[The size of token2index dictionary is {len(self.tok2ind)}]")
        logger.debug(
            f"[The size of index2token dictionary is {len(self.ind2tok)}]")

    def _load_other_data(self):
        # opendialkg
        self.entity2id = json.load(
            open(os.path.join(self.dpath, 'entity2id.json'), encoding='utf-8'))  # {entity: entity_id}
        self.id2entity = {idx: entity for entity,
                          idx in self.entity2id.items()}
        self.n_entity = max(self.entity2id.values()) + 1
        # {head_entity_id: [(relation_id, tail_entity_id)]}
        self.entity_kg = open(os.path.join(
            self.dpath, 'opendialkg_subkg.txt'), encoding='utf-8')
        logger.debug(
            f"[Load entity dictionary and KG from {os.path.join(self.dpath, 'opendialkg_subkg.json')} and {os.path.join(self.dpath, 'opendialkg_triples.txt')}]")

        # conceptnet
        # {concept: concept_id}
        self.word2id = json.load(
            open(os.path.join(self.dpath, 'word2id.json'), 'r', encoding='utf-8'))
        self.n_word = max(self.word2id.values()) + 1
        # {concept \t relation\t concept}
        self.word_kg = open(os.path.join(
            self.dpath, 'concept_subkg.txt'), encoding='utf-8')
        logger.debug(
            f"[Load word dictionary and KG from {os.path.join(self.dpath, 'word2id.json')} and {os.path.join(self.dpath, 'concept_subkg.txt')}]")

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
        augmented_convs = [self._convert_to_id(
            conversation) for conversation in tqdm(raw_data)]
        augmented_conv_dicts = []
        for conv in tqdm(augmented_convs):
            augmented_conv_dicts.extend(self._augment_and_add(conv))
        return augmented_conv_dicts

    def _convert_to_id(self, conversation):
        augmented_convs = []
        last_role = None
        for utt in conversation['dialog']:
            text_token_ids = [self.tok2ind.get(
                word, self.unk_token_idx) for word in utt["text"]]
            item_ids = [self.entity2id[movie]
                        for movie in utt['item'] if movie in self.entity2id]
            entity_ids = [self.entity2id[entity]
                          for entity in utt['entity'] if entity in self.entity2id]
            word_ids = [self.word2id[word]
                        for word in utt['word'] if word in self.word2id]

            if utt["role"] == last_role:
                augmented_convs[-1]["text"] += text_token_ids
                augmented_convs[-1]["item"] += item_ids
                augmented_convs[-1]["entity"] += entity_ids
                augmented_convs[-1]["word"] += word_ids
            else:
                augmented_convs.append({
                    "role": utt["role"],
                    "text": text_token_ids,
                    "entity": entity_ids,
                    "item": item_ids,
                    "word": word_ids
                })
            last_role = utt["role"]

        return augmented_convs

    def _augment_and_add(self, raw_conv_dict):
        augmented_conv_dicts = []
        context_tokens, context_entities, context_words, context_items = [], [], [], []
        entity_set, word_set = set(), set()
        for i, conv in enumerate(raw_conv_dict):
            text_tokens, entities, items, words = conv["text"], conv["entity"], conv["item"], conv["word"]
            if len(context_tokens) > 0:
                conv_dict = {
                    'role': conv['role'],
                    "context_tokens": copy(context_tokens),
                    "response": text_tokens,
                    "context_entities": copy(context_entities),
                    "context_words": copy(context_words),
                    'context_items': copy(context_items),
                    "items": items,
                }
                augmented_conv_dicts.append(conv_dict)

            context_tokens.append(text_tokens)
            context_items += items
            for entity in entities + items:
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
        item_entity_ids = json.load(
            open(os.path.join(self.dpath, 'item_ids.json'), 'r', encoding='utf-8'))
        logger.debug('[Load item entity ids]')

        side_data = {
            "entity_kg": processed_entity_kg,
            "word_kg": processed_word_kg,
            "item_entity_ids": item_entity_ids,
        }
        return side_data

    def _entity_kg_process(self):
        edge_list = []  # [(entity, entity, relation)]
        for line in self.entity_kg:
            triple = line.strip().split('\t')
            if len(triple) != 3 or triple[0] not in self.entity2id or triple[2] not in self.entity2id:
                continue
            e0 = self.entity2id[triple[0]]
            e1 = self.entity2id[triple[2]]
            r = triple[1]
            edge_list.append((e0, e1, r))
            # edge_list.append((e1, e0, r))
            edge_list.append((e0, e0, 'SELF_LOOP'))
            if e1 != e0:
                edge_list.append((e1, e1, 'SELF_LOOP'))

        relation_cnt, relation2id, edges, entities = defaultdict(
            int), dict(), set(), set()
        for h, t, r in edge_list:
            relation_cnt[r] += 1
        for h, t, r in edge_list:
            if relation_cnt[r] > 20000:
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
            triple = line.strip().split('\t')
            entities.add(triple[0])
            entities.add(triple[2])
            e0 = self.word2id[triple[0]]
            e1 = self.word2id[triple[2]]
            edges.add((e0, e1))
            edges.add((e1, e0))
        # edge_set = [[co[0] for co in list(edges)], [co[1] for co in list(edges)]]
        return {
            'edge': list(edges),
            'entity': list(entities)
        }

    def split_text(self, data):

        all_data = []
        for each in tqdm(data):
            each_dict = {}
            each_data = []
            for one in each['dialog']:
                str_text = one['text']
                list_text = self.Tokenizer.tokenize(str_text)
                one['text'] = list_text
                each_data.append(one)
            each_dict['dialog'] = each_data
            all_data.append(each_dict)

        return all_data

    def generate_tok2ind(self, processed_train_data):

        cnt = 0
        tok2ind = {}

        if self.tokenize == 'nltk' or self.tokenize == 'jieba':
            tok2ind['__pad__'] = cnt
            cnt += 1
            tok2ind['__start__'] = cnt
            cnt += 1
            tok2ind['__end__'] = cnt
            cnt += 1
            tok2ind['__unk__'] = cnt
            cnt += 1
        elif self.tokenize == 'bert':
            tok2ind['[PAD]'] = cnt
            cnt += 1

        for i in tqdm(processed_train_data):
            dialog = i['dialog']
            for each_dialog in dialog:
                text = each_dialog['text']
                for each_word in text:
                    if each_word not in tok2ind:
                        tok2ind[each_word] = cnt
                        cnt += 1

        if self.tokenize == 'nltk':
            tok2ind['_split_'] = cnt
            cnt += 1

        return tok2ind

    def generate_copy_mask(self, tok2ind, processed_train_data):

        copy_mask = np.zeros((len(tok2ind)), dtype=bool)
        for each_data in tqdm(processed_train_data):
            for dialog in each_data['dialog']:
                match_list = []
                text = dialog['text']
                for word in dialog['word']:
                    list_word = self.Tokenizer.tokenize(word)
                    match_list += list_word
                for entity in dialog['entity']:
                    list_word = self.Tokenizer.tokenize(entity)
                    match_list += list_word

                for item in dialog['item']:
                    list_word = self.Tokenizer.tokenize(item)
                    match_list += list_word

                match_list = list(set(match_list))

                for each_word in text:
                    if each_word in match_list:
                        token_id = tok2ind[each_word]
                        copy_mask[token_id] = True

        return copy_mask

    def generate_word2vec(self, processed_train_data):

        corpus = []
        for each_data in processed_train_data:
            for dialog in each_data['dialog']:
                text = dialog['text']
                corpus.append(text)

        model = gensim.models.word2vec.Word2Vec(
            corpus, vector_size=300, min_count=1)

        if self.tokenize == 'nltk':
            word2index = {word: i + 4 for i,
                          word in enumerate(model.wv.index_to_key)}
            word2embedding = [[0] * 300] * 4 + [model.wv[word]
                                                for word in word2index] + [[0] * 300]

        elif self.tokenize == 'jieba':
            word2index = {word: i + 4 for i,
                          word in enumerate(model.wv.index_to_key)}
            word2embedding = [[0] * 300] * 4 + [model.wv[word]
                                                for word in word2index]

        return word2embedding
