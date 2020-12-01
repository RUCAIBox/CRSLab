# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/23, 2020/12/1
# @Author : Kun Zhou, Xiaolei Wang
# @Email  : francis_kun_zhou@163.com, wxl1999@foxmail.com

from copy import deepcopy

import torch
from tqdm import tqdm

from crslab.data.dataloader.base_dataloader import BaseDataLoader, padded_tensor, get_onehot_label, truncate, \
    merge_utt


class KGSFDataLoader(BaseDataLoader):
    def __init__(self, opt, dataset):
        self.n_entity = opt['n_entity']
        self.pad_token_idx = opt['pad_token_idx']
        self.pad_entity_idx = opt['pad_entity_idx']
        self.pad_word_idx = opt['pad_word_idx']
        self.context_truncate = opt.get('context_truncate', None)
        self.response_truncate = opt.get('response_truncate', None)
        self.entity_truncate = opt.get('entity_truncate', None)
        self.word_truncate = opt.get('word_truncate', None)
        super().__init__(opt, dataset)

    def get_pretrain_data(self, batch_size, shuffle=True):
        """
        input: conv_dict = {
                    "context_tokens": [id1, id2, ..., ],  # [int]
                    "context_entities": [id1, id2, ..., ],  # [int]
                    "context_words": [id1, id2, ..., ],  # [int]
                    "response": [id1, id2, ..., ],  # [int]
                    "items": [id1, id2, ..., ],  # [int]
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
                    "items": id,  # int
                }
        output: torch.tensors (context_words, movie)
        """
        batch_context_entities = []
        batch_context_words = []
        for conv_dict in batch:
            batch_context_entities.append(conv_dict['context_entities'])
            batch_context_words.append(conv_dict['context_words'])

        return (padded_tensor(batch_context_words, self.pad_word_idx),
                get_onehot_label(batch_context_entities, self.n_entity))

    def rec_process_fn(self):
        """
        Sometimes, the recommender may recommend more than one movies to seeker,
        hence we need to augment data for each recommended movie
        """
        augment_dataset = []
        for conv_dict in tqdm(self.dataset):
            for movie in conv_dict['items']:
                augment_conv_dict = deepcopy(conv_dict)
                augment_conv_dict['item'] = movie
                augment_dataset.append(augment_conv_dict)
        return augment_dataset

    def rec_batchify(self, batch):
        """
        input: conv_dict = {
                    "context_tokens": [id1, id2, ..., ],  # [int]
                    "context_entities": [id1, id2, ..., ],  # [int]
                    "context_words": [id1, id2, ..., ],  # [int]
                    "response": [id1, id2, ..., ],  # [int]
                    "item": id,  # int
                }
        output: torch.tensors (context_entities, context_words, movie)
        """
        batch_context_entities = []
        batch_context_words = []
        batch_movie = []
        for conv_dict in batch:
            batch_context_entities.append(conv_dict['context_entities'])
            batch_context_words.append(conv_dict['context_words'])
            batch_movie.append(conv_dict['item'])

        return (padded_tensor(batch_context_entities, self.pad_entity_idx),
                padded_tensor(batch_context_words, self.pad_word_idx),
                get_onehot_label(batch_context_entities, self.n_entity),
                torch.tensor(batch_movie, dtype=torch.long))

    def conv_batchify(self, batch):
        """
        input: conv_dict = {
                    "context_tokens": [id1, id2, ..., ],  # [int]
                    "context_entities": [id1, id2, ..., ],  # [int]
                    "context_words": [id1, id2, ..., ],  # [int]
                    "response": [id1, id2, ..., ],  # [int]
                    "items": [id1, id2, ..., ],  # [int]
                }
        output: torch.tensors (context_tokens, context_entities, context_words, response)
        """
        batch_context_tokens = []
        batch_context_entities = []
        batch_context_words = []
        batch_response = []
        for conv_dict in batch:
            batch_context_tokens.append(truncate(merge_utt(conv_dict['context_tokens']), self.context_truncate, truncate_tail=False))
            batch_context_entities.append(truncate(conv_dict['context_entities'], self.entity_truncate))
            batch_context_words.append(truncate(conv_dict['context_words'], self.word_truncate))
            batch_response.append(truncate(conv_dict['response'], self.response_truncate))

        return (padded_tensor(batch_context_tokens, self.pad_token_idx, right_padded=False),
                padded_tensor(batch_context_entities, self.pad_entity_idx),
                padded_tensor(batch_context_words, self.pad_word_idx),
                padded_tensor(batch_response, self.pad_token_idx))

    def guide_batchify(self, *args, **kwargs):
        pass
