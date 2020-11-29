# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/23, 2020/11/29
# @Author : Kun Zhou, Xiaolei Wang
# @Email  : francis_kun_zhou@163.com, wxl1999@foxmail.com

import json
import os
import pickle as pkl
from collections import defaultdict
from copy import copy

from loguru import logger
from tqdm import tqdm

from crslab.data.dataset.base_dataset import BaseDataset, DATA_PATH
from crslab.data.dataset.download import DownloadableFile, build


class ReDialDataset(BaseDataset):
    def __init__(self, opt, restore=False, save=False):
        self.n_entity = opt.get('n_entity', 0)
        self.n_word = opt.get('n_word', 0)
        self.context_truncate = opt.get('context_truncate', None)
        self.response_truncate = opt.get('response_truncate', None)
        self.entity_truncate = opt.get('entity_truncate', None)
        self.word_truncate = opt.get('word_truncate', None)

        dpath = os.path.join(DATA_PATH, "redial")
        super().__init__(opt, dpath, restore, save)

    def _load_vocab(self):
        self.pad_token = '__pad__'
        self.start_token = '__start__'
        self.end_token = '__end__'
        self.unk_token = '__unk__'
        self.pad_token_idx = 0
        self.start_token_idx = 1
        self.end_token_idx = 2
        self.unk_token_idx = 3

        tok2ind = json.load(open(os.path.join(self.dpath, "word2index_redial.json"), 'r', encoding='utf-8'))
        tok2ind.update({self.pad_token: self.pad_token_idx,
                        self.start_token: self.start_token_idx,
                        self.end_token: self.end_token_idx,
                        self.unk_token: self.unk_token_idx})
        ind2tok = {idx: word for word, idx in tok2ind.items()}

        logger.info("[Load vocab from {}]", os.path.join(self.dpath, "word2index_redial.json"))
        logger.debug("[The size of token2index dictionary is {}]", len(tok2ind))
        logger.debug("[The size of index2token dictionary is {}]", len(ind2tok))

        return tok2ind, ind2tok

    def _load_data(self):
        """
        load raw data and necessary side information for preprocessing
        raw: train_data/valid_data/test_data;
        side:
        """

        # download
        dfile = DownloadableFile('1FykgR_dSkARAggGIuyckZ5ZbT9EMLi3c', 'redial.zip',
                                 '7978ee519c3c347e8107135102a120e6c161c881698195dc076fe694fad47c0f',
                                 from_google=True)
        build(self.dpath, dfile)

        # load train/valid/test data
        with open(os.path.join(self.dpath, "train_data_reformulated.json"), 'r', encoding='utf-8') as f:
            train_data = json.load(f)
            logger.info("[Load train data from {}]", "train_data_reformulated.json")
        with open(os.path.join(self.dpath, "valid_data_reformulated.json"), 'r', encoding='utf-8') as f:
            valid_data = json.load(f)
            logger.info("[Load valid data from {}]", "valid_data_reformulated.json")
        with open(os.path.join(self.dpath, "test_data_reformulated.json"), 'r', encoding='utf-8') as f:
            test_data = json.load(f)
            logger.info("[Load test data from {}]", "test_data_reformulated.json")

        # create dictionary: tok2ind, ind2tok
        tok2ind, ind2tok = self._load_vocab()

        # dbpedia
        self.entity2id = pkl.load(open(os.path.join(self.dpath, "entity2entityId.pkl"), 'rb'))  # {entity: entity_id}
        # {head_entity_id: [(relation_id, tail_entity_id)]}
        self.entity_kg = pkl.load(open(os.path.join(self.dpath, "subkg.pkl"), 'rb'))
        logger.info("[Load entity dictionary and KG data from {} and {}]",
                    "entity2entityId.pkl", "subkg.pkl")

        # conceptNet
        # {concept: concept_id}
        self.word2id = json.load(open(os.path.join(self.dpath, "key2index_3rd.json"), 'r', encoding='utf-8'))
        # {relation\t concept \t concept}
        self.word_kg = open(os.path.join(self.dpath, "conceptnet_subkg.txt"), 'r', encoding='utf-8')
        logger.info("[Load word dictionary and KG data from {} and {}]",
                    "key2index_3rd.json", "conceptnet_subkg.txt")

        return train_data, valid_data, test_data, tok2ind, ind2tok

    def _data_preprocess(self, train_data, valid_data, test_data):
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
        processed_train_data = self._raw_data_process(train_data)
        logger.info("[Finish train data process]")
        processed_valid_data = self._raw_data_process(valid_data)
        logger.info("[Finish valid data process]")
        processed_test_data = self._raw_data_process(test_data)
        logger.info("[Finish test data process]")
        processed_side_data = self._side_data_process()
        logger.info("[Finish side data process]")
        return processed_train_data, processed_valid_data, processed_test_data, processed_side_data

    def _raw_data_process(self, raw_data):
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
        augmented_convs = [self._merge_conv_data(conversation["dialog"]) for conversation in tqdm(raw_data)]
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
                    "context_tokens": self.truncate(copy(context_tokens), self.context_truncate, truncate_tail=False),
                    "context_entities": self.truncate(copy(context_entities), self.entity_truncate),
                    "context_words": self.truncate(copy(context_words), self.word_truncate),
                    "response": self.truncate(copy(response_add_SE), self.response_truncate),
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

    def _side_data_process(self):
        processed_entity_kg = self._entity_kg_process()
        logger.info("[Finish entity KG preprocess]")
        processed_word_kg = self._word_kg_process()
        logger.info("[Finish word KG preprocess]")
        movie_entity_ids = pkl.load(open(os.path.join(self.dpath, 'movie_ids.pkl'), 'rb'))
        logger.info('[Load movie entity ids]')
        side_data = {
            "entity_kg": processed_entity_kg,
            "word_kg": processed_word_kg,
            "movie_entity_id": movie_entity_ids,
        }
        return side_data

    def _entity_kg_process(self, SELF_LOOP_ID=185):
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

    def _word_kg_process(self):
        """
        return [(head_word, tail_word)]
        """
        edges = set()  # {(entity, entity)}
        for line in self.word_kg:
            kg = line.strip().split('\t')
            e0 = self.word2id[kg[1].split('/')[0]]
            e1 = self.word2id[kg[2].split('/')[0]]
            edges.add((e0, e1))
            edges.add((e1, e0))
        # edge_set = [[co[0] for co in list(edges)], [co[1] for co in list(edges)]]
        return list(edges)
