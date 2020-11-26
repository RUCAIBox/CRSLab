# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/23, 2020/11/26
# @Author : Kun Zhou, Xiaolei Wang
# @Email  : francis_kun_zhou@163.com, wxl1999@foxmail.com

import json
import os
import pickle as pkl
from collections import defaultdict
from copy import copy

from loguru import logger
from tqdm import tqdm

from crslab.data.dataset.base_dataset import BaseDataset


class ReDialDataset(BaseDataset):
    def __init__(self, config, restore=False, save=False):
        super().__init__(config, restore, save)

    def _load_token_dictionary(self):
        self.tok2ind = json.load(open(os.path.join(self.data_path, self.config['data_files']['token2index']),
                                      'r', encoding='utf-8'))
        self.pad_token = '__pad__'
        self.start_token = '__start__'
        self.end_token = '__end__'
        self.unk_token = '__unk__'
        self.pad_token_idx = self.config['pad_token_idx']
        self.start_token_idx = self.config['start_token_idx']
        self.end_token_idx = self.config['end_token_idx']
        self.unk_token_idx = self.config['unk_token_idx']
        self.tok2ind.update({self.pad_token: self.pad_token_idx,
                             self.start_token: self.start_token_idx,
                             self.end_token: self.end_token_idx,
                             self.unk_token: self.unk_token_idx})
        self.ind2tok = {idx: word for word, idx in self.tok2ind.items()}
        logger.info("[Load token dictionary from {}]", self.config['data_files']['token2index'])
        logger.debug("[The size of token2index dictionary is {}]", len(self.tok2ind))
        logger.debug("[The size of index2token dictionary is {}]", len(self.ind2tok))

    def _load_data(self):
        """
        load raw data and necessary side information for preprocessing
        raw: train_data/valid_data/test_data;
        side:
        """
        if not os.path.exists(os.path.join(self.data_path, self.config['data_files']['train_data'])) or \
                not os.path.exists(os.path.join(self.data_path, self.config['data_files']['valid_data'])) or \
                not os.path.exists(os.path.join(self.data_path, self.config['data_files']['test_data'])):
            raise FileNotFoundError

        # load train/valid/test data
        with open(os.path.join(self.data_path, self.config['data_files']['train_data']), 'r', encoding='utf-8') as f:
            self.train_data = json.load(f)
            logger.info("[Load train data from {}]", self.config['data_files']['train_data'])
        with open(os.path.join(self.data_path, self.config['data_files']['valid_data']), 'r', encoding='utf-8') as f:
            self.valid_data = json.load(f)
            logger.info("[Load valid data from {}]", self.config['data_files']['valid_data'])
        with open(os.path.join(self.data_path, self.config['data_files']['test_data']), 'r', encoding='utf-8') as f:
            self.test_data = json.load(f)
            logger.info("[Load test data from {}]", self.config['data_files']['test_data'])

        # create dictionary: tok2ind, ind2tok
        self._load_token_dictionary()

        # dbpedia: {entity: entity_id}
        self.entity2id = pkl.load(open(os.path.join(self.data_path, self.config['data_files']['entity2id']), 'rb'))
        # [movie_entity_id]
        # self.movie_entity_id = pkl.load(open(os.path.join(self.data_path, "movie_ids.pkl"), 'rb'))
        self.n_entity = self.config['n_entity']
        # dbpedia: {head_entity_id: [(relation_id, tail_entity_id)]}
        self.entity_kg = pkl.load(open(os.path.join(self.data_path, self.config['data_files']['entity_kg']), 'rb'))
        logger.info("[Load entity dictionary and KG data from {} and {}]",
                    self.config['data_files']['entity2id'], self.config['data_files']['entity_kg'])

        # {concept: concept_id}
        self.word2id = json.load(
            open(os.path.join(self.data_path, self.config['data_files']['word2id']), 'r', encoding='utf-8'))
        self.n_word = self.config['n_word']
        # {relation\t concept \t concept}
        self.word_kg = open(os.path.join(self.data_path, self.config['data_files']['word_kg']), 'r', encoding='utf-8')
        logger.info("[Load word dictionary and KG data from {} and {}]",
                    self.config['data_files']['word2id'], self.config['data_files']['word_kg'])

    def _data_preprocess(self):
        """
        raw_data_output: {
            'dialog_context': the preprocessed contextual dialog;
            'interaction_context': if necessary, the preprocessed interaction history;
            'entity_context': if necessary, the entities in context;
            'word_context': if necessary, the words in context;
            'rec_item': the recommended item in this turn;
            'response': the ground-truth response;
        }
        side_data_output: {
            'entity_knowledge_graph': if necessary, entity knowledge graph as side information;
            'word_knowledge_graph': if necessary, word knowledge graph as side information;}
        """
        processed_train_data = self._raw_data_preprocessing(self.train_data)
        logger.info("[Finish train data preprocess]")
        processed_valid_data = self._raw_data_preprocessing(self.valid_data)
        logger.info("[Finish valid data preprocess]")
        processed_test_data = self._raw_data_preprocessing(self.test_data)
        logger.info("[Finish test data preprocess]")

        # side information
        processed_side_data = self._side_data_preprocessing()
        return processed_train_data, processed_valid_data, processed_test_data, processed_side_data

    def _raw_data_preprocessing(self, raw_data):
        """
        raw_data_output: {
            'dialog_context': the preprocessed contextual dialog;
            'interaction_context': if necessary, the preprocessed interaction history;
            'entity_context': if necessary, the entities in context;
            'word_context': if necessary, the words in context;
            'rec_item': the recommended item in this turn;
            'response': the ground-truth response;
        }
        """
        augmented_convs = [self._merge_conv_data(conversation["dialog"]) for conversation in raw_data]
        augmented_conv_dicts = []
        for conv in tqdm(augmented_convs):
            augmented_conv_dicts.extend(self._augment_and_truncate(conv))
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
            movie_ids = [self.entity2id[movie] for movie in utt['movies'] if movie in self.entity2id]
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

    def _augment_and_truncate(self, raw_conv_dict):
        """
        input: {
                    "role": role,
                    "text": text_token_ids,
                    "entity": entity_ids,
                    "movie": movie_ids,
                    "word": word_ids
                }
        1.augment one conversation into several instances;
        2.pad or truncate;
        """
        augmented_conv_dicts = []
        context_tokens, context_entities, context_words = [], [], []
        entity_set, word_set = set(), set()
        for i, conv in enumerate(raw_conv_dict):
            text_tokens, entities, movies, words = conv["text"], conv["entity"], conv["movie"], conv["word"]
            # to id
            if len(context_tokens) > 0:
                response_add_SE = self.add_start_end_token_idx(text_tokens, add_start=True,
                                                               start_idx=self.start_token_idx, add_end=True,
                                                               end_idx=self.end_token_idx)
                conv_dict = {
                    "context_tokens": self.truncate(context_tokens, self.config['max_length'], truncate_head=True),
                    "context_entities": self.truncate(context_entities, self.config['max_length']),
                    "context_words": self.truncate(context_words, self.config['max_length']),
                    "response": self.truncate(response_add_SE, self.config['max_length']),
                    "movie": copy(movies)
                }
                augmented_conv_dicts.append(conv_dict)

            context_tokens += text_tokens
            for entity in entities + movies:
                if entity not in entity_set:
                    entity_set.add(entity)
                    context_entities.append(entity)
            for word in words:
                if word not in word_set:
                    word_set.add(word)
                    context_words.append(word)

        return augmented_conv_dicts

    def _side_data_preprocessing(self):
        processed_entity_kg = self._entity_kg_preprocess()
        logger.info("[Finish entity KG preprocess]")
        processed_word_kg = self._word_kg_preprocess()
        logger.info("[Finish word KG preprocess]")
        return processed_entity_kg, processed_word_kg

    def _entity_kg_preprocess(self, SELF_LOOP_ID=185):
        """get dbpedia edge information

        Args:

        Returns:
            list: edge list [(head_entity_id, tail_entity_id, new_relation_id)]
        """
        edge_list = []  # [(entity, entity, relation)]
        for entity in range(self.n_entity):
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

    def _word_kg_preprocess(self):
        """
        return [(head_word, tail_word)]
        """
        edges = set()  # {(entity, entity)}
        for line in self.word_kg:
            kg = line.strip().split('\t')
            entity0 = self.word2id[kg[1].split('/')[0]]
            entity1 = self.word2id[kg[2].split('/')[0]]
            edges.add((entity0, entity1))
            edges.add((entity1, entity0))
        # edge_set = [[co[0] for co in list(edges)], [co[1] for co in list(edges)]]
        return list(edges)
