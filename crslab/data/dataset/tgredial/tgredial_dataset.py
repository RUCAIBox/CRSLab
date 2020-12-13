# @Time   : 2020/12/4
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/12/6, 2020/12/8
# @Author : Kun Zhou, Xiaolei Wang
# @Email  : francis_kun_zhou@163.com

import json
import os
from collections import defaultdict
from copy import copy

from loguru import logger
from tqdm import tqdm

from crslab.config.config import DATA_PATH
from crslab.data.dataset.base_dataset import BaseDataset
from .resource import resources


class TGReDialDataset(BaseDataset):
    def __init__(self, opt, tokenize, restore=False, save=False):
        resource = resources[tokenize]
        self.special_token_idx = resource['special_token_idx']
        self.unk_token_idx = self.special_token_idx['unk']
        self.pad_topic_idx = self.special_token_idx['pad_topic']
        dpath = os.path.join(DATA_PATH, 'tgredial', tokenize)
        super().__init__(opt, dpath, resource, restore, save)

    def _load_vocab(self):
        self.tok2ind = json.load(open(os.path.join(self.dpath, 'token2id.json'), 'r', encoding='utf-8'))
        self.ind2tok = {idx: word for word, idx in self.tok2ind.items()}

        logger.debug(f"[Load vocab from {os.path.join(self.dpath, 'token2id.json')}]")
        logger.debug(f"[The size of token2index dictionary is {len(self.tok2ind)}]")
        logger.debug(f"[The size of index2token dictionary is {len(self.ind2tok)}]")

    def _load_data(self):
        # load train/valid/test data
        with open(os.path.join(self.dpath, 'train_data.json'), 'r', encoding='utf-8') as f:
            train_data = json.load(f)
            logger.debug(f"[Load train data from {os.path.join(self.dpath, 'train_data.json')}]")
        with open(os.path.join(self.dpath, 'valid_data.json'), 'r', encoding='utf-8') as f:
            valid_data = json.load(f)
            logger.debug(f"[Load valid data from {os.path.join(self.dpath, 'valid_data.json')}]")
        with open(os.path.join(self.dpath, 'test_data.json'), 'r', encoding='utf-8') as f:
            test_data = json.load(f)
            logger.debug(f"[Load test data from {os.path.join(self.dpath, 'test_data.json')}]")

        self._load_vocab()

        # cn-dbpedia
        self.entity2id = json.load(
            open(os.path.join(self.dpath, 'entity2id.json'), encoding='utf-8'))  # {entity: entity_id}
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

        # topic dictionary
        self.topic2id = json.load(open(os.path.join(self.dpath, 'topic2id.json'), 'r', encoding='utf-8'))
        self.n_topic = len(self.topic2id) + 1
        logger.debug(f"[Load topic dictionary from {os.path.join(self.dpath, 'topic2id.json')}]")

        # user interaction history dictionary
        self.conv2history = json.load(open(os.path.join(self.dpath, 'user2history.json'), 'r', encoding='utf-8'))
        logger.debug(f"[Load user interaction history from {os.path.join(self.dpath, 'user2history.json')}]")

        # user profile
        self.user2profile = json.load(open(os.path.join(self.dpath, 'user2profile.json'), 'r', encoding='utf-8'))
        logger.debug(f"[Load user profile from {os.path.join(self.dpath, 'user2profile.json')}")

        vocab = {
            'tok2ind': self.tok2ind,
            'ind2tok': self.ind2tok,
            'vocab_size': len(self.tok2ind),
            'n_entity': self.n_entity,
            'n_word': self.n_word,
            'n_topic': self.n_topic
        }
        vocab.update(self.special_token_idx)

        return train_data, valid_data, test_data, vocab

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
                    'entity' (list of str): mentioned entities of cn-dbpedia in text
                    'word' (list of str): mentioned words of hownet in text
                    'target' (list of str): topic in current turn
                    'final' (list): final goal for current turn
                }
            }

        Returns:
            list of dict: {
                'role' (str): 'Seeker' or 'Recommender';
                'user_profile' (list of list of int): id of tokens of sentences of user profile
                'context_tokens' (list of list int): token ids of the preprocessed contextual dialog;
                'response' (list of int): token ids of the ground-truth response;
                'interaction_history' (list of int): id of items which have interaction of the user in current turn;
                'items' (list of int): item ids mentioned in current turn, we only keep those in dbpedia for comparison;
                'context_entities' (list of int): if necessary, id of entities in context;
                'context_words' (list of int): if necessary, id of words in context;
                'context_policy' (list of list of list): policy of each context turn, ont turn may have several policies, where first is action and second is keyword;
                'target' (list): policy of current turn;
                'final' (list): final goal for current turn;
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
        for utt in conversation['messages']:
            assert utt['role'] != last_role

            text_token_ids = [self.tok2ind.get(word, self.unk_token_idx) for word in utt["text"]]
            movie_ids = [self.entity2id[movie] for movie in utt['movie'] if movie in self.entity2id]
            entity_ids = [self.entity2id[entity] for entity in utt['entity'] if entity in self.entity2id]
            word_ids = [self.word2id[word] for word in utt['word'] if word in self.word2id]
            policy = []
            for action, kw in zip(utt['target'][1::2], utt['target'][2::2]):
                if kw is None or action == '推荐电影':
                    policy.append([action, self.pad_topic_idx])
                    continue
                if isinstance(kw, str):
                    kw = [kw]
                kw = [self.topic2id.get(k, self.pad_topic_idx) for k in kw]
                policy.append([action, kw])
            final_kws = [self.topic2id[kw] if kw is not None else self.pad_topic_idx for kw in utt['final'][1]]
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
        """
        augment one conversation into several instances;
        """
        augmented_conv_dicts = []
        context_tokens, context_entities, context_words, context_policy, context_items = [], [], [], [], []
        entity_set, word_set = set(), set()
        for i, conv in enumerate(raw_conv_dict):
            text_tokens, entities, movies, words, policies = conv["text"], conv["entity"], conv["movie"], conv["word"], \
                                                             conv['policy']
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
        movie_entity_ids = json.load(open(os.path.join(self.dpath, 'movie_ids.json'), 'r', encoding='utf-8'))
        logger.debug('[Load movie entity ids]')

        side_data = {
            "entity_kg": processed_entity_kg,
            "word_kg": processed_word_kg,
            "item_entity_ids": movie_entity_ids,
        }
        return side_data

    def _entity_kg_process(self):
        """get cn-dbpedia edge information

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

        relation_cnt, relation2id, edges = defaultdict(int), dict(), set()
        for h, t, r in edge_list:
            relation_cnt[r] += 1
        for h, t, r in edge_list:
            if relation_cnt[r] > 1000:
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
