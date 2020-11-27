# @Time   : 2020/11/27
# @Author : Xiaolei Wang
# @Email  : wxl1999@foxmail.com

# UPDATE:
# @Time   : 2020/11/27
# @Author : Xiaolei Wang
# @Email  : wxl1999@foxmail.com

from copy import deepcopy

import torch

from crslab.data.dataloader.base_dataloader import BaseDataLoader


class KBRDDataLoader(BaseDataLoader):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

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
        output: torch.tensors (context_entities, movie)
        """
        context_entities = []
        movies = []
        for conv_dict in batch:
            context_entities.append(conv_dict['context_entities'])
            movies.append(conv_dict['movie'])

        return (self.padded_tensor(context_entities, self.config['entity_pad']),
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
        output: torch.tensors (context_tokens, context_entities, response)
        """
        context_tokens = []
        context_entities = []
        response = []
        for conv_dict in batch:
            context_tokens.append(conv_dict['context_tokens'])
            context_entities.append(conv_dict['context_entities'])
            response.append(conv_dict['response'])

        return (self.padded_tensor(context_tokens, self.config['pad_token_idx'], right_padded=False),
                self.padded_tensor(context_entities, self.config['entity_pad']),
                self.padded_tensor(response, self.config['pad_token_idx']))
