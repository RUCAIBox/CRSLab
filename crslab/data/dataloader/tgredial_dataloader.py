# @Time   : 2020/12/9
# @Author : Yuanhang Zhou
# @Email  : sdzyh002@gmail.com

# UPDATE:
# @Time   : 2020/12/14
# @Author : Xiaolei Wang
# @Email  : wxl1999@foxmail.com

import random
from copy import deepcopy

import torch
from tqdm import tqdm

from crslab.data.dataloader.base_dataloader import BaseDataLoader
from crslab.data.dataloader.utils import add_start_end_token_idx, padded_tensor, truncate, merge_utt


class TGReDialDataLoader(BaseDataLoader):
    def __init__(self, opt, dataset, vocab):
        super().__init__(opt, dataset)

        self.n_entity = vocab['n_entity']
        self.item_size = self.n_entity
        self.pad_token_idx = vocab['pad']
        self.start_token_idx = vocab['start']
        self.end_token_idx = vocab['end']
        self.unk_token_idx = vocab['unk']
        self.pad_entity_idx = vocab['pad_entity']
        self.pad_word_idx = vocab['pad_word']
        self.sent_split_idx = vocab['sent_split']
        self.word_split_idx = vocab['word_split']
        self.conv_bos_id = vocab['start']

        self.context_truncate = opt.get('context_truncate', None)
        self.response_truncate = opt.get('response_truncate', None)
        self.entity_truncate = opt.get('entity_truncate', None)
        self.word_truncate = opt.get('word_truncate', None)
        self.item_truncate = opt.get('item_truncate', None)

    def rec_process_fn(self, *args, **kwargs):
        augment_dataset = []
        for conv_dict in tqdm(self.dataset):
            for movie in conv_dict['items']:
                augment_conv_dict = deepcopy(conv_dict)
                augment_conv_dict['item'] = movie
                augment_dataset.append(augment_conv_dict)
        return augment_dataset

    def _process_rec_context(self, context_tokens):
        """convert the context form dataset format to dataloader format

        Args:
            context_tokens: list of list of int, no padding and no adding special token

        Returns:
            compat_context: list of int, padding and adding special token
                [CLS, ID1, ID2, sent_split_idx, ID3, ID4, ..., SEP]
        """
        compat_context = [
            utter + [self.sent_split_idx] for utter in context_tokens
        ]
        compat_context = compat_context[:-1]
        compat_context = truncate(merge_utt(compat_context),
                                  self.context_truncate - 2,
                                  truncate_tail=False)
        compat_context = add_start_end_token_idx(compat_context, True,
                                                 self.start_token_idx, True,
                                                 self.end_token_idx)
        return compat_context

    def _neg_sample(self, item_set):
        item = random.randint(1, self.item_size)
        while item in item_set:
            item = random.randint(1, self.item_size)
        return item

    def _process_history(self, context_items, item_id):
        """
        Args:
            context_items: 历史交互记录
            item_id: 要预测的item id
        """
        input_ids = truncate(context_items,
                             max_length=self.item_truncate,
                             truncate_tail=False)
        target_pos = input_ids[1:] + [item_id]
        input_mask = [1] * len(input_ids)

        sample_negs = []
        seq_set = set(input_ids)
        for _ in input_ids:
            sample_negs.append(self._neg_sample(seq_set))

        return input_ids, target_pos, input_mask, sample_negs

    def rec_batchify(self, batch):
        batch_context = []
        batch_movie_id = []
        batch_input_ids = []
        batch_target_pos = []
        batch_input_mask = []
        batch_sample_negs = []

        for conv_dict in batch:
            context = self._process_rec_context(conv_dict['context_tokens'])
            batch_context.append(context)

            item_id = conv_dict['item']
            batch_movie_id.append(item_id)

            context_items = conv_dict['context_items']
            input_ids, target_pos, input_mask, sample_negs = self._process_history(
                context_items, item_id)
            batch_input_ids.append(input_ids)
            batch_target_pos.append(target_pos)
            batch_input_mask.append(input_mask)
            batch_sample_negs.append(sample_negs)

        batch_context = padded_tensor(batch_context,
                                      self.pad_token_idx,
                                      max_len=self.context_truncate)
        batch_mask = (batch_context != 0).long()

        return (batch_context, batch_mask,
                padded_tensor(batch_input_ids,
                              pad_idx=self.pad_token_idx,
                              pad_tail=False,
                              max_len=self.item_truncate),
                padded_tensor(batch_target_pos,
                              pad_idx=self.pad_token_idx,
                              pad_tail=False,
                              max_len=self.item_truncate),
                padded_tensor(batch_input_mask,
                              pad_idx=self.pad_token_idx,
                              pad_tail=False,
                              max_len=self.item_truncate),
                padded_tensor(batch_sample_negs,
                              pad_idx=self.pad_token_idx,
                              pad_tail=False,
                              max_len=self.item_truncate),
                torch.tensor(batch_movie_id))

    def conv_batchify(self, batch):
        """collate batch data for conversation

        Args:
            batch (list of dict):

        Returns:
            response: [BOS, TOKEN, TOKEN, EOS] length=response_truncate
        """
        batch_context_tokens = []
        batch_response = []
        # batch_masks = []
        batch_context_entities = []
        batch_context_words = []
        for conv_dict in batch:
            context_tokens = [utter + [self.conv_bos_id] for utter in conv_dict['context_tokens']]
            context_tokens[-1] = context_tokens[-1][:-1]
            batch_context_tokens.append(
                truncate(merge_utt(context_tokens), max_length=self.context_truncate, truncate_tail=False),
            )
            batch_response.append(
                add_start_end_token_idx(
                    truncate(conv_dict['response'], max_length=self.response_truncate - 2),
                    add_start=True,
                    start_token_idx=self.start_token_idx,
                    add_end=True,
                    end_token_idx=self.end_token_idx
                )
            )
            batch_context_entities.append(
                truncate(conv_dict['context_entities'],
                         self.entity_truncate,
                         truncate_tail=False))
            batch_context_words.append(
                truncate(conv_dict['context_words'],
                         self.word_truncate,
                         truncate_tail=False))
        batch_context_tokens = padded_tensor(items=batch_context_tokens,
                                             pad_idx=self.pad_token_idx,
                                             max_len=self.context_truncate,
                                             pad_tail=False)
        batch_response = padded_tensor(batch_response,
                                       pad_idx=self.pad_token_idx,
                                       max_len=self.response_truncate,
                                       pad_tail=True)
        batch_input_ids = torch.cat((batch_context_tokens, batch_response), dim=1)

        return (
            batch_input_ids,
            batch_context_tokens,
            padded_tensor(batch_context_entities,
                          self.pad_entity_idx,
                          pad_tail=False),
            padded_tensor(batch_context_words,
                          self.pad_word_idx,
                          pad_tail=False),
            batch_response
        )

    def policy_batchify(self, batch):
        pass
