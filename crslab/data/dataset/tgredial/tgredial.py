# @Time   : 2020/12/4
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/12/6, 2021/1/2, 2020/12/19
# @Author : Kun Zhou, Xiaolei Wang, Yuanhang Zhou
# @Email  : francis_kun_zhou@163.com, sdzyh002@gmail

# UPDATE
# @Time    :   2022/9/26
# @Author  :   Xinyu Tang
# @email   :   txy20010310@163.com

r"""
TGReDial
========
References:
    Zhou, Kun, et al. `"Towards Topic-Guided Conversational Recommender System."`_ in COLING 2020.

.. _`"Towards Topic-Guided Conversational Recommender System."`:
   https://www.aclweb.org/anthology/2020.coling-main.365/

"""

import json
import os
from collections import defaultdict
from copy import copy
import numpy as np
import gensim

from loguru import logger
from tqdm import tqdm

from crslab.config import DATASET_PATH, BERT_ZH_PATH, GPT2_ZH_PATH, MODEL_PATH
from crslab.data.dataset.base import BaseDataset
from .resources import resources
from crslab.data.dataset.tokenize import CrsTokenize


class TGReDialDataset(BaseDataset):
    """

    Attributes:
        train_data: train dataset.
        valid_data: valid dataset.
        test_data: test dataset.
        vocab (dict): ::

            {
                'tok2ind': map from token to index,
                'ind2tok': map from index to token,
                'topic2ind': map from topic to index,
                'ind2topic': map from index to topic,
                'entity2id': map from entity to index,
                'id2entity': map from index to entity,
                'word2id': map from word to index,
                'vocab_size': len(self.tok2ind),
                'n_topic': len(self.topic2ind) + 1,
                'n_entity': max(self.entity2id.values()) + 1,
                'n_word': max(self.word2id.values()) + 1,
            }

    Notes:
        ``'unk'`` and ``'pad_topic'`` must be specified in ``'special_token_idx'`` in ``resources.py``.

    """

    def __init__(self, opt, tokenize, restore=False, save=False):
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
        resource = resources['resource']
        token = resource[tokenize]
        self.special_token_idx = token['special_token_idx']
        self.unk_token_idx = self.special_token_idx['unk']
        self.pad_topic_idx = self.special_token_idx['pad_topic']

        self.tokenize = tokenize
        self.path = None
        if tokenize == 'bert':
            self.path = BERT_ZH_PATH
        elif tokenize == 'gpt2':
            self.path = GPT2_ZH_PATH
        self.crstokenizer = CrsTokenize('zh', tokenize, self.path)
        dpath = os.path.join(DATASET_PATH, 'tgredial')

        self.replace_token = opt.get('replace_token',None)
        self.replace_token_idx = opt.get('replace_token_idx',None)
        super().__init__(opt, dpath, resource, restore, save)
        if self.replace_token:
            if self.replace_token_idx:
                self.side_data["embedding"][self.replace_token_idx] = self.side_data['embedding'][0]
            else:
                self.side_data["embedding"] = np.insert(self.side_data["embedding"],len(self.side_data["embedding"]),self.side_data['embedding'][0],axis=0)
        

    def _load_data(self):
        train_data, valid_data, test_data = self._load_raw_data()
        self._load_vocab()
        self._load_other_data()

        vocab = {
            'tok2ind': self.tok2ind,
            'ind2tok': self.ind2tok,
            'topic2ind': self.topic2ind,
            'ind2topic': self.ind2topic,
            'entity2id': self.entity2id,
            'id2entity': self.id2entity,
            'word2id': self.word2id,
            'vocab_size': len(self.tok2ind),
            'n_topic': len(self.topic2ind) + 1,
            'n_entity': self.n_entity,
            'n_word': self.n_word,
        }
        vocab.update(self.special_token_idx)

        return train_data, valid_data, test_data, vocab

    def _load_raw_data(self):
        # load train/valid/test data
        with open(os.path.join(self.dpath, 'train_data.json'), 'r', encoding='utf-8') as f:
            train_data = json.load(f)
            logger.debug(f"[Load train data from {os.path.join(self.dpath, 'train_data.json')}]")
        # split token
        processing_train_data = self.split_token(train_data)
        logger.info("[Finish train data split]")
        # generate tok2ind
        tok2ind = self.generate_tok2ind(processing_train_data)
        logger.info("[Finish generate train tok2ind]")
        # generate word2vec
        self.generate_word2vec(processing_train_data)
        logger.info('[Finish generate word2vec]')
        # build copy_mask
        if self.copy:
            copy_mask = self.generate_copy_mask(tok2ind, processing_train_data)
            logger.info('[Finish generate copy_mask]')

        with open(os.path.join(self.dpath, 'valid_data.json'), 'r', encoding='utf-8') as f:
            valid_data = json.load(f)
            logger.debug(f"[Load valid data from {os.path.join(self.dpath, 'valid_data.json')}]")
        # split_token
        processing_valid_data = self.split_token(valid_data)
        logger.info("[Finish valid data split]")

        with open(os.path.join(self.dpath, 'test_data.json'), 'r', encoding='utf-8') as f:
            test_data = json.load(f)
            logger.debug(f"[Load test data from {os.path.join(self.dpath, 'test_data.json')}]")
        # split_token
        processing_test_data = self.split_token(test_data)
        logger.info("[Finish test data split]")

        return processing_train_data, processing_valid_data, processing_test_data

    def _load_vocab(self):
        self.tok2ind = json.load(open(os.path.join(self.dpath, 'token2id.json'), 'r', encoding='utf-8'))
        self.ind2tok = {idx: word for word, idx in self.tok2ind.items()}
        # add special tokens
        if self.replace_token:
            if self.replace_token not in self.tok2ind:
                if self.replace_token_idx:
                    self.ind2tok[self.replace_token_idx] = self.replace_token
                    self.tok2ind[self.replace_token] = self.replace_token_idx
                    self.special_token_idx[self.replace_token] = self.replace_token_idx
                else:
                    self.ind2tok[len(self.tok2ind)] = self.replace_token
                    self.tok2ind[self.replace_token] = len(self.tok2ind)
                    self.special_token_idx[self.replace_token] = len(self.tok2ind)-1 
        logger.debug(f"[Load vocab from {os.path.join(self.dpath, 'token2id.json')}]")
        logger.debug(f"[The size of token2index dictionary is {len(self.tok2ind)}]")
        logger.debug(f"[The size of index2token dictionary is {len(self.ind2tok)}]")

        self.topic2ind = json.load(open(os.path.join(self.dpath, 'topic2id.json'), 'r', encoding='utf-8'))
        self.ind2topic = {idx: word for word, idx in self.topic2ind.items()}

        logger.debug(f"[Load vocab from {os.path.join(self.dpath, 'topic2id.json')}]")
        logger.debug(f"[The size of token2index dictionary is {len(self.topic2ind)}]")
        logger.debug(f"[The size of index2token dictionary is {len(self.ind2topic)}]")

    def _load_other_data(self):
        # cn-dbpedia
        self.entity2id = json.load(
            open(os.path.join(self.dpath, 'entity2id.json'), encoding='utf-8'))  # {entity: entity_id}
        self.id2entity = {idx: entity for entity, idx in self.entity2id.items()}
        self.n_entity = max(self.entity2id.values()) + 1
        # {head_entity_id: [(relation_id, tail_entity_id)]}
        self.entity_kg = open(os.path.join(self.dpath, 'cn-dbpedia.txt'), encoding='utf-8')
        logger.debug(
            f"[Load entity dictionary and KG from {os.path.join(self.dpath, 'entity2id.json')} and {os.path.join(self.dpath, 'cn-dbpedia.txt')}]")

        # hownet
        # {concept: concept_id}
        self.word2id = json.load(open(os.path.join(self.dpath, 'word2id.json'), 'r', encoding='utf-8'))
        self.n_word = max(self.word2id.values()) + 1
        # {relation\t concept \t concept}
        self.word_kg = open(os.path.join(self.dpath, 'hownet.txt'), encoding='utf-8')
        logger.debug(
            f"[Load word dictionary and KG from {os.path.join(self.dpath, 'word2id.json')} and {os.path.join(self.dpath, 'hownet.txt')}]")

        # user interaction history dictionary
        self.conv2history = json.load(open(os.path.join(self.dpath, 'user2history.json'), 'r', encoding='utf-8'))
        logger.debug(f"[Load user interaction history from {os.path.join(self.dpath, 'user2history.json')}]")

        # user profile
        self.user2profile = json.load(open(os.path.join(self.dpath, 'user2profile.json'), 'r', encoding='utf-8'))
        logger.debug(f"[Load user profile from {os.path.join(self.dpath, 'user2profile.json')}")


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
        augmented_convs = [self._convert_to_id(conversation) for conversation in tqdm(raw_data)]
        augmented_conv_dicts = []
        for conv in tqdm(augmented_convs):
            augmented_conv_dicts.extend(self._augment_and_add(conv))
        return augmented_conv_dicts

    def _convert_to_id(self, conversation):
        augmented_convs = []
        last_role = None
        for utt in conversation['messages']:
            assert utt['role'] != last_role
            # change movies into slots
            if self.replace_token:
                if len(utt['movie']) != 0:
                    while  '《' in utt['text'] :
                        begin = utt['text'].index("《")
                        end = utt['text'].index("》")
                        utt['text'] = utt['text'][:begin] + [self.replace_token] + utt['text'][end+1:]
            text_token_ids = [self.tok2ind.get(word, self.unk_token_idx) for word in utt["text"]]
            movie_ids = [self.entity2id[movie] for movie in utt['movie'] if movie in self.entity2id]
            entity_ids = [self.entity2id[entity] for entity in utt['entity'] if entity in self.entity2id]
            word_ids = [self.word2id[word] for word in utt['word'] if word in self.word2id]
            policy = []
            for action, kw in zip(utt['target'][1::2], utt['target'][2::2]):
                if kw is None or action == '推荐电影':
                    continue
                if isinstance(kw, str):
                    kw = [kw]
                kw = [self.topic2ind.get(k, self.pad_topic_idx) for k in kw]
                policy.append([action, kw])
            final_kws = [self.topic2ind[kw] if kw is not None else self.pad_topic_idx for kw in utt['final'][1]]
            final = [utt['final'][0], final_kws]
            conv_utt_id = str(conversation['conv_id']) + '/' + str(utt['local_id'])
            interaction_history = self.conv2history.get(conv_utt_id, [])
            user_profile = self.user2profile[conversation['user_id']]
            user_profile = [[self.tok2ind.get(token, self.unk_token_idx) for token in sent] for sent in user_profile]

            augmented_convs.append({
                "role": utt["role"],
                "text": text_token_ids,
                "entity": entity_ids,
                "movie": movie_ids,
                "word": word_ids,
                'policy': policy,
                'final': final,
                'interaction_history': interaction_history,
                'user_profile': user_profile
            })
            last_role = utt["role"]

        return augmented_convs

    def _augment_and_add(self, raw_conv_dict):
        augmented_conv_dicts = []
        context_tokens, context_entities, context_words, context_policy, context_items = [], [], [], [], []
        entity_set, word_set = set(), set()
        for i, conv in enumerate(raw_conv_dict):
            text_tokens, entities, movies, words, policies = conv["text"], conv["entity"], conv["movie"], conv["word"], \
                                                             conv['policy']
            if self.replace_token is not None: 
                if text_tokens.count(30000) != len(movies):
                    continue # the number of slots doesn't equal to the number of movies
                
            if len(context_tokens) > 0:
                conv_dict = {
                    'role': conv['role'],
                    'user_profile': conv['user_profile'],
                    "context_tokens": copy(context_tokens),
                    "response": text_tokens,
                    "context_entities": copy(context_entities),
                    "context_words": copy(context_words),
                    'interaction_history': conv['interaction_history'],
                    'context_items': copy(context_items),
                    "items": movies,
                    'context_policy': copy(context_policy),
                    'target': policies,
                    'final': conv['final'],
                }
                augmented_conv_dicts.append(conv_dict)

            context_tokens.append(text_tokens)
            context_policy.append(policies)
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
        movie_entity_ids = json.load(open(os.path.join(self.dpath, 'movie_ids.json'), 'r', encoding='utf-8'))
        logger.debug('[Load movie entity ids]')

        side_data = {
            "entity_kg": processed_entity_kg,
            "word_kg": processed_word_kg,
            "item_entity_ids": movie_entity_ids,
        }
        return side_data

    def _entity_kg_process(self):
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

        relation_cnt, relation2id, edges, entities = defaultdict(int), dict(), set(), set()
        for h, t, r in edge_list:
            relation_cnt[r] += 1
        for h, t, r in edge_list:
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

    def split_token(self, data):
        
        all_data = []
        for each in tqdm(data):
            each_dict = {}
            each_data = []
            each_dict['conv_id'] = each['conv_id']
            for one in each['messages']:
                str_text = one['text']
                tokenizer = self.tokenize
                crstokenize = self.crstokenizer
                list_text = crstokenize.tokenize(str_text, tokenizer)
                one['text'] = list_text
                each_data.append(one)
            each_dict['messages'] = each_data
            each_dict['user_id'] = each['user_id']
            all_data.append(each_dict)
        
        return all_data

    def generate_tok2ind(self, processed_train_data):

        cnt = 0
        tok2ind = {}

        if self.tokenize == 'nltk' or self.tokenize == 'jieba' or self.tokenize == 'pkuseg':
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
            dialog = i['messages']
            for each_dialog in dialog:
                text = each_dialog['text']
                for each_word in text:
                    if each_word not in tok2ind:
                        tok2ind[each_word] = cnt
                        cnt += 1
        
        if self.tokenize == 'nltk':
            tok2ind['_split_'] = cnt
            cnt += 1

        tok2ind_path = os.path.join(DATASET_PATH, 'tgredial', 'token2id.json')
        with open(tok2ind_path, 'w', encoding='utf-8') as write:
            json.dump(tok2ind, write, ensure_ascii=False, indent=4, separators=(',', ':'))

        return tok2ind

    def generate_copy_mask(self, tok2ind, processing_train_data):
        
        tokenizer = self.tokenize
        crstokenize = self.crstokenizer

        copy_mask = np.zeros((len(tok2ind)), dtype=bool)
        for each_data in tqdm(processing_train_data):
            for dialog in each_data['messages']:
                match_list = []
                text = dialog['text']
                for word in dialog['word']:
                    list_word = crstokenize.tokenize(word, tokenizer)
                    match_list += list_word

                for movie in dialog['movie']:
                    list_word = crstokenize.tokenize(movie, tokenizer)
                    match_list += list_word

                for entity in dialog['entity']:
                    list_word = crstokenize.tokenize(entity, tokenizer)
                    match_list += list_word
                    
                match_list = list(set(match_list))
                
                for each_word in text:
                    if each_word in match_list:
                        token_id = tok2ind[each_word]
                        copy_mask[token_id] = True

        if not os.path.exists(MODEL_PATH):
            os.mkdir(MODEL_PATH)

        if not os.path.exists(os.path.join(MODEL_PATH, 'kgsf')):
            os.mkdir(os.path.join(MODEL_PATH, 'kgsf'))

        copy_mask_dirpath = os.path.join(MODEL_PATH, 'kgsf', 'TGReDial')
        if not os.path.exists(copy_mask_dirpath):
            os.mkdir(copy_mask_dirpath)

        path = os.path.join(MODEL_PATH, 'kgsf', 'TGReDial', 'copy_mask.npy')
        np.save(path, copy_mask)


    def generate_word2vec(self, processing_train_data):

        corpus = []
        for each_data in processing_train_data:
            for dialog in each_data['messages']:
                text = dialog['text']
                corpus.append(text)

        model = gensim.models.word2vec.Word2Vec(corpus, vector_size=300, min_count=1)

        if self.tokenize == 'nltk':
            word2index = {word: i + 4 for i, word in enumerate(model.wv.index_to_key)}        
            word2embedding = [[0] * 300] * 4 + [model.wv[word] for word in word2index] + [[0] * 300]

        elif self.tokenize == 'jieba' or self.tokenize == 'pkuseg':
            word2index = {word: i + 4 for i, word in enumerate(model.wv.index_to_key)}        
            word2embedding = [[0] * 300] * 4 + [model.wv[word] for word in word2index]            

        elif self.tokenize == 'bert':
            word2index = {word: i + 1 for i, word in enumerate(model.wv.index_to_key)}        
            word2embedding = [[0] * 300] + [model.wv[word] for word in word2index]

        elif self.tokenize == 'gpt2':
            word2index = {word: i + 1 for i, word in enumerate(model.wv.index_to_key)}        
            word2embedding = [model.wv[word] for word in word2index]

        word2vec_path = os.path.join(DATASET_PATH, 'tgredial', 'word2vec.npy')
        np.save(word2vec_path, word2embedding)
