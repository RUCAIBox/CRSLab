# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/23, 2020/11/26
# @Author : Kun Zhou, Xiaolei Wang
# @Email  : francis_kun_zhou@163.com, wxl1999@foxmail.com

import os
import pickle as pkl
import torch
from copy import deepcopy

from crslab.data.dataloader.base_dataloader import BaseDataLoader


class KGSFDataLoader(BaseDataLoader):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

    def get_pretrain_data(self, batch_size, shuffle=False):
        """
        input: conv_dict = {
                    "context_tokens": [id1, id2, ..., ],  # [int]
                    "context_entities": [id1, id2, ..., ],  # [int]
                    "context_words": [id1, id2, ..., ],  # [int]
                    "response": [id1, id2, ..., ],  # [int]
                    "movie": [id1, id2, ..., ],  # [int]
                }
        output:
            a list: [batch1, batch2, ... ],
            each batch is the input for model (context_entities, context_words);
        """
        return self.get_data(self.pretrain_batchify, batch_size, shuffle)

    def pretrain_batchify(self, batch):
        """
        input: conv_dict = {
                    "context_tokens": [id1, id2, ..., ],  # [int]
                    "context_entities": [id1, id2, ..., ],  # [int]
                    "context_words": [id1, id2, ..., ],  # [int]
                    "response": [id1, id2, ..., ],  # [int]
                    "movie": id,  # int
                }
        output: torch.tensors (context_words, movie)
        """
        context_entities = []
        context_words = []
        for conv_dict in batch:
            context_entities.append(conv_dict['context_entities'])
            context_words.append(conv_dict['context_words'])

        return (self.padded_tensor(context_words, self.config['word_pad']),
                self.get_onehot_label(context_entities, self.config['n_entity']))

    @property
    def rec_process_fn(self):
        """
        Sometimes, the recommender may recommend more than one movies to seeker,
        hence we need to augment data for each recommended movie
        """
        augment_dataset = []
        for conv_dict in self.dataset:
            for movie in conv_dict['movie']:
                augment_conv_dict = deepcopy(conv_dict)
                augment_conv_dict['movie'] = movie
                augment_dataset.append(augment_conv_dict)
        return augment_dataset

    def rec_batchify(self, batch):
        """
        input: conv_dict = {
                    "context_tokens": [id1, id2, ..., ],  # [int]
                    "context_entities": [id1, id2, ..., ],  # [int]
                    "context_words": [id1, id2, ..., ],  # [int]
                    "response": [id1, id2, ..., ],  # [int]
                    "movie": id,  # int
                }
        output: torch.tensors (context_entities, context_words, movie)
        """
        context_entities = []
        context_words = []
        movies = []
        for conv_dict in batch:
            context_entities.append(conv_dict['context_entities'])
            context_words.append(conv_dict['context_words'])
            movies.append(conv_dict['movie'])

        return (self.padded_tensor(context_entities, self.config['entity_pad']),
                self.padded_tensor(context_words, self.config['word_pad']),
                self.get_onehot_label(context_entities, self.config['n_entity']),
                torch.tensor(movies, dtype=torch.long))

    def conv_batchify(self, batch):
        """
        input: conv_dict = {
                    "context_tokens": [id1, id2, ..., ],  # [int]
                    "context_entities": [id1, id2, ..., ],  # [int]
                    "context_words": [id1, id2, ..., ],  # [int]
                    "response": [id1, id2, ..., ],  # [int]
                    "movie": [id1, id2, ..., ],  # [int]
                }
        output: torch.tensors (context_tokens, context_entities, context_words, response)
        """
        context_tokens = []
        context_entities = []
        context_words = []
        response = []
        for conv_dict in batch:
            context_tokens.append(conv_dict['context_tokens'])
            context_entities.append(conv_dict['context_entities'])
            context_words.append(conv_dict['context_words'])
            response.append(conv_dict['response'])

        return (self.padded_tensor(context_tokens, self.config['pad_token_idx'], right_padded=False),
                self.padded_tensor(context_entities, self.config['entity_pad']),
                self.padded_tensor(context_words, self.config['word_pad']),
                self.padded_tensor(response, self.config['pad_token_idx']))

    def get_movie_ids(self):
        movie_ids_path = os.path.join(self.config['data_path'], 'movie_ids.pkl')
        return pkl.load(open(movie_ids_path, 'rb'))
