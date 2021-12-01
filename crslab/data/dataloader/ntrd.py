# @Time   : 2021/10/06
# @Author : Zhipeng Zhao
# @Email  : oran_official@outlook.com

from copy import deepcopy

import torch
from tqdm import tqdm

from crslab.data.dataloader.base import BaseDataLoader
from crslab.data.dataloader.utils import add_start_end_token_idx, merge_utt_replace, padded_tensor, get_onehot, truncate, merge_utt


class NTRDDataLoader(BaseDataLoader):
    def __init__(self, opt, dataset, vocab):
        """

        Args:
            opt (Config or dict): config for dataloader or the whole system.
            dataset: data for model.
            vocab (dict): all kinds of useful size, idx and map between token and idx.

        """
        super().__init__(opt, dataset)
        self.n_entity = vocab['n_entity']
        self.pad_token_idx = vocab['pad']
        self.start_token_idx = vocab['start']
        self.end_token_idx = vocab['end']
        self.pad_entity_idx = vocab['pad_entity']
        self.pad_word_idx = vocab['pad_word']
        self.context_truncate = opt.get('context_truncate', None)
        self.response_truncate = opt.get('response_truncate', None)
        self.entity_truncate = opt.get('entity_truncate', None)
        self.word_truncate = opt.get('word_truncate', None)
        self.replace_token = opt.get('replace_token',None)
        self.replace_token_idx = vocab[self.replace_token]

    def get_pretrain_data(self, batch_size, shuffle=True):
        return self.get_data(self.pretrain_batchify, batch_size, shuffle, self.retain_recommender_target)

    def pretrain_batchify(self, batch):
        batch_context_entities = []
        batch_context_words = []
        for conv_dict in batch:
            batch_context_entities.append(
                truncate(conv_dict['context_entities'], self.entity_truncate, truncate_tail=False))
            batch_context_words.append(truncate(conv_dict['context_words'], self.word_truncate, truncate_tail=False))

        return (padded_tensor(batch_context_words, self.pad_word_idx, pad_tail=False),
                get_onehot(batch_context_entities, self.n_entity))

    def rec_process_fn(self):
        augment_dataset = []
        for conv_dict in tqdm(self.dataset):
            if conv_dict['role'] == 'Recommender':
                for movie in conv_dict['items']:
                    augment_conv_dict = deepcopy(conv_dict)
                    augment_conv_dict['item'] = movie
                    augment_dataset.append(augment_conv_dict)
        return augment_dataset

    def rec_batchify(self, batch):
        batch_context_entities = []
        batch_context_words = []
        batch_item = []
        for conv_dict in batch:
            batch_context_entities.append(
                truncate(conv_dict['context_entities'], self.entity_truncate, truncate_tail=False))
            batch_context_words.append(truncate(conv_dict['context_words'], self.word_truncate, truncate_tail=False))
            batch_item.append(conv_dict['item'])

        return (padded_tensor(batch_context_entities, self.pad_entity_idx, pad_tail=False),
                padded_tensor(batch_context_words, self.pad_word_idx, pad_tail=False),
                get_onehot(batch_context_entities, self.n_entity),
                torch.tensor(batch_item, dtype=torch.long))

    def conv_process_fn(self, *args, **kwargs):
        return self.retain_recommender_target()

    def conv_batchify(self, batch):
        batch_context_tokens = []
        batch_context_entities = []
        batch_context_words = []
        batch_response = []
        flag = False
        batch_all_movies = [] 
        for conv_dict in batch:
            temp = add_start_end_token_idx(truncate(conv_dict['response'], self.response_truncate - 2),
                                        start_token_idx=self.start_token_idx,
                                        end_token_idx=self.end_token_idx)

            if temp.count(self.replace_token_idx) != 0:
                flag = True
            batch_context_tokens.append(
                truncate(merge_utt(conv_dict['context_tokens']), self.context_truncate, truncate_tail=False))
            batch_context_entities.append(
                truncate(conv_dict['context_entities'], self.entity_truncate, truncate_tail=False))
            batch_context_words.append(truncate(conv_dict['context_words'], self.word_truncate, truncate_tail=False))
            batch_response.append(
                add_start_end_token_idx(truncate(conv_dict['response'], self.response_truncate - 2),
                                        start_token_idx=self.start_token_idx,
                                        end_token_idx=self.end_token_idx))
            
            batch_all_movies.append(
                truncate(conv_dict['items'], temp.count(self.replace_token_idx), truncate_tail=False)) #only use movies, not all entities.
        if flag == False:# zero slot in a batch
            return False

        return (padded_tensor(batch_context_tokens, self.pad_token_idx, pad_tail=False),
                padded_tensor(batch_context_entities, self.pad_entity_idx, pad_tail=False),
                padded_tensor(batch_context_words, self.pad_word_idx, pad_tail=False),
                padded_tensor(batch_response, self.pad_token_idx),
                padded_tensor(batch_all_movies, self.pad_entity_idx, pad_tail=False)) 

    def policy_batchify(self, *args, **kwargs):
        pass