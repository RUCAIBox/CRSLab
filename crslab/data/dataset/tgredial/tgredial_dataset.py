# @Time   : 2020/12/4
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/12/6, 2020/12/6
# @Author : Kun Zhou, Xiaolei Wang
# @Email  : francis_kun_zhou@163.com

import json
import os
from collections import defaultdict
from copy import copy

from loguru import logger
from tqdm import tqdm

from crslab.config.config import DATA_PATH
from crslab.data.dataloader.base_dataloader import add_start_end_token_idx
from crslab.data.dataset.base_dataset import BaseDataset
from crslab.data.dataset.download import DownloadableFile, build
from .resource import resources


class TGReDialDataset(BaseDataset):
    def __init__(self, opt, restore=False, save=False):
        tokenize = opt.get('tokenize', 'pkuseg')
        resource = resources[tokenize]

        dpath = os.path.join(DATA_PATH, 'tgredial', tokenize)
        dfile = resource['file']
        build(dpath, dfile, version=resource['version'])
        super().__init__(opt, dpath, restore, save)

    def _load_vocab(self):
        self.pad_token = '__pad__'
        self.start_token = '__start__'
        self.end_token = '__end__'
        self.unk_token = '__unk__'

        tok2ind = json.load(open(os.path.join(self.dpath, self.tok2ind_file), 'r', encoding='utf-8'))
        tok2ind.update({self.pad_token: self.pad_token_idx,
                        self.start_token: self.start_token_idx,
                        self.end_token: self.end_token_idx,
                        self.unk_token: self.unk_token_idx})
        ind2tok = {idx: word for word, idx in tok2ind.items()}

        logger.debug(f"[Load vocab from {self.tok2ind_file}]")
        logger.debug(f"[The size of token2index dictionary is {len(tok2ind)}]")
        logger.debug(f"[The size of index2token dictionary is {len(ind2tok)}]")

        return tok2ind, ind2tok

    def _load_data(self):
        # download
        dfile = DownloadableFile('1q3ipHbZy6erCldC22dZSkQeUrQXTTzID', 'tg-redial.zip',
                                 '5b1e5159b1af3ed9bc42183b8c445be170c45d5d0e5cd5d1bf8bc45c7ad7dca2',
                                 from_google=True)

        # load train/valid/test data
        with open(os.path.join(self.dpath, self.train_data_file), 'r', encoding='utf-8') as f:
            train_data = json.load(f)
            logger.debug(f"[Load train data from {self.train_data_file}]")
        with open(os.path.join(self.dpath, self.valid_data_file), 'r', encoding='utf-8') as f:
            valid_data = json.load(f)
            logger.debug(f"[Load valid data from {self.valid_data_file}]")
        with open(os.path.join(self.dpath, self.test_data_file), 'r', encoding='utf-8') as f:
            test_data = json.load(f)
            logger.debug(f"[Load test data from {self.test_data_file}]")

        # load dictionary: tok2ind, ind2tok
        tok2ind, ind2tok = self._load_vocab()

        # cn-dbpedia
        self.entity2id = json.load(
            open(os.path.join(self.dpath, "entity2id.json"), encoding='utf-8'))  # {entity: entity_id}
        # {head_entity_id: [(relation_id, tail_entity_id)]}
        self.entity_kg = [triple.strip().split('\t') for triple in
                          open(os.path.join(self.dpath, "cn-dbpedia.txt"), encoding='utf-8')]
        logger.debug(f"[Load entity dictionary and KG from {'entity2id.json'} and {'subkg_2nd.txt'}]")

        # hownet
        # {concept: concept_id}
        self.word2id = json.load(open(os.path.join(self.dpath, "word2id.json"), 'r', encoding='utf-8'))
        # {relation\t concept \t concept}
        self.word_kg = open(os.path.join(self.dpath, "hownet.txt"), encoding='utf-8')
        logger.debug(f"[Load word dictionary and KG from {'word2id.json'} and {'hownet.txt'}]")

        # topic dictionary
        self.topic2id = json.load(open(os.path.join(self.dpath, "topic2id.json"), 'r', encoding='utf-8'))
        logger.debug(f"[Load topic dictionary from {'topic2id.json'}]")

        # user interaction history dictionary
        self.user_utt2history = json.load(open(os.path.join(self.dpath, "user2history.json"), 'r', encoding='utf-8'))
        logger.debug(f"[Load user id and utterance to history dictionary from {'user2history.json'}]")

        return train_data, valid_data, test_data, tok2ind, ind2tok

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
                'dialog' (list of dict): {
                    'utt_id' (int): id of current utterance
                    'role' (str): 'Seeker' or 'Recommender'
                    'text' (list of str): utterance which has benn tokenized into tokens
                    'movies' (list of str): mentioned movies in text
                    'entity' (list of str): mentioned entities of dbpedia in text
                    'word' (list of str): mentioned words of conceptnet in text
                }
            }

        Returns:
            list of dict: {
                'context_tokens' (list of list int): token ids of the preprocessed contextual dialog;
                'response' (list of int): token ids of the ground-truth response;
                'items' (list of int): item ids mentioned in current turn, we only keep those in dbpedia for comparison;
                'context_entities' (list of int): if necessary, id of entities in context;
                'context_words' (list of int): if necessary, id of words in context;
            }
        """

        augmented_convs = [self._merge_conv_data(conversation["messages"]) for conversation in tqdm(raw_data)]
        augmented_conv_dicts = []
        for conv in tqdm(augmented_convs):
            augmented_conv_dicts.extend(self._augment_and_add(conv))
        return augmented_conv_dicts

    def _merge_conv_data(self, dialog):
        """
        1.merge the continue utterances based on roles;
        2.convert token/word/entity/movie into ids;
        """
        # {"utt_id": 0,
        # "role": "Seeker",
        # "text": ["Hi", "I", "am", "looking", "for", "a", "movie", "like", "@111776"],
        # "movies": ["<http://dbpedia.org/resource/Super_Troopers>"],
        # "entity": [],
        # "word": ["am", "looking", "for", "a", "movie", "like", "@111776"]}
        augmented_convs = []
        last_role = None
        for utt in dialog:
            text_token_ids = [self.tok2ind.get(word, self.unk_token_idx) for word in utt["text"]]
            movie_ids = [self.entity2id[movie] for movie in utt['movie'] if movie in self.entity2id]
            entity_ids = [self.entity2id[entity] for entity in utt['entity'] if entity in self.entity2id]
            word_ids = [self.word2id[word] for word in utt['word'] if word in self.word2id]

            if utt["role"] == last_role:
                augmented_convs[-1]["text"] += text_token_ids
                augmented_convs[-1]["movie"] += movie_ids
                augmented_convs[-1]["entity"] += entity_ids
                augmented_convs[-1]["word"] += word_ids
            else:
                augmented_convs.append({
                    "role": utt["role"],
                    "text": text_token_ids,
                    "entity": entity_ids,
                    "movie": movie_ids,
                    "word": word_ids
                })
            last_role = utt["role"]

        return augmented_convs

    def _augment_and_add(self, raw_conv_dict):
        """
        1.augment one conversation into several instances;
        2.add start or end token;
        input: {
            "role": role,
            "text": text_token_ids,
            "entity": entity_ids,
            "movie": movie_ids,
            "word": word_ids
        }
        """
        augmented_conv_dicts = []
        context_tokens, context_entities, context_words = [], [], []
        entity_set, word_set = set(), set()
        for i, conv in enumerate(raw_conv_dict):
            text_tokens, entities, movies, words = conv["text"], conv["entity"], conv["movie"], conv["word"]
            if len(context_tokens) > 0:
                response_add_SE = add_start_end_token_idx(text_tokens, add_start=True,
                                                          start_token_idx=self.start_token_idx, add_end=True,
                                                          end_token_idx=self.end_token_idx)
                conv_dict = {
                    'role': conv['role'],
                    "context_tokens": copy(context_tokens),
                    "context_entities": copy(context_entities),
                    "context_words": copy(context_words),
                    "response": response_add_SE,
                    "items": movies
                }
                augmented_conv_dicts.append(conv_dict)

            context_tokens.append(text_tokens)
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
        # movie_entity_ids = pkl.load(open(os.path.join(self.dpath, 'movie_entity_ids.pkl'), 'rb'))
        # logger.debug('[Load movie entity ids]')

        side_data = {
            "entity_kg": processed_entity_kg,
            "word_kg": processed_word_kg,
            # "item_entity_ids": movie_entity_ids,
        }
        return side_data

    def _entity_kg_process(self, SELF_LOOP_ID=185):
        """get cn-dbpedia edge information

        Args:

        Returns:
            list: edge list [(head_entity_id, tail_entity_id, new_relation_id)]
        """
        edge_list = []  # [(entity, entity, relation)]
        for entity in tqdm(range(self.n_entity)):
            # add self loop
            edge_list.append((entity, entity, SELF_LOOP_ID))
            if entity not in self.entity_kg:
                continue
            for tail_and_relation in self.entity_kg[entity]:
                if entity != tail_and_relation[1] and tail_and_relation[0] != SELF_LOOP_ID:
                    edge_list.append((entity, tail_and_relation[1], tail_and_relation[0]))
                    edge_list.append((tail_and_relation[1], entity, tail_and_relation[0]))

        relation_cnt, relation2id, edges = defaultdict(int), dict(), set()
        for h, t, r in edge_list:
            relation_cnt[r] += 1
        for h, t, r in edge_list:
            if relation_cnt[r] > 1000:
                if r not in relation2id:
                    relation2id[r] = len(relation2id)
                edges.add((h, t, relation2id[r]))
        return list(edges)

    def _word_kg_process(self):
        """return [(head_word, tail_word)]"""
        edges = set()  # {(entity, entity)}
        for line in tqdm(self.word_kg):
            kg = line.strip().split('\t')
            e0 = self.word2id[kg[0]]
            e1 = self.word2id[kg[2]]
            edges.add((e0, e1))
            edges.add((e1, e0))
        # edge_set = [[co[0] for co in list(edges)], [co[1] for co in list(edges)]]
        return list(edges)
