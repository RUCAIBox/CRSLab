# @Time   : 2020/12/9
# @Author : Yuanhang Zhou
# @Email  : sdzyh002@gmail.com

# UPDATE:
# @Time   : 2020/12/29, 2020/12/15
# @Author : Xiaolei Wang, Yuanhang Zhou
# @Email  : wxl1999@foxmail.com, sdzyh002@gmail

import random
from copy import deepcopy

import torch
from tqdm import tqdm

from crslab.data.dataloader.base import BaseDataLoader
from crslab.data.dataloader.utils import add_start_end_token_idx, padded_tensor, truncate, merge_utt


class TGReDialDataLoader(BaseDataLoader):
    """Dataloader for model TGReDial.

    Notes:
        You can set the following parameters in config:

        - ``'context_truncate'``: the maximum length of context.
        - ``'response_truncate'``: the maximum length of response.
        - ``'entity_truncate'``: the maximum length of mentioned entities in context.
        - ``'word_truncate'``: the maximum length of mentioned words in context.
        - ``'item_truncate'``: the maximum length of mentioned items in context.

        The following values must be specified in ``vocab``:

        - ``'pad'``
        - ``'start'``
        - ``'end'``
        - ``'unk'``
        - ``'pad_entity'``
        - ``'pad_word'``

        the above values specify the id of needed special token.

        - ``'ind2tok'``: map from index to token.
        - ``'tok2ind'``: map from token to index.
        - ``'vocab_size'``: size of vocab.
        - ``'id2entity'``: map from index to entity.
        - ``'n_entity'``: number of entities in the entity KG of dataset.
        - ``'sent_split'`` (optional): token used to split sentence. Defaults to ``'end'``.
        - ``'word_split'`` (optional): token used to split word. Defaults to ``'end'``.
        - ``'pad_topic'`` (optional): token used to pad topic.
        - ``'ind2topic'`` (optional): map from index to topic.

    """

    def __init__(self, opt, dataset, vocab):
        """

        Args:
            opt (Config or dict): config for dataloader or the whole system.
            dataset: data for model.
            vocab (dict): all kinds of useful size, idx and map between token and idx.

        """
        super().__init__(opt, dataset)

        self.n_entity = vocab['n_entity']
        self.item_size = self.n_entity
        self.pad_token_idx = vocab['pad']
        self.start_token_idx = vocab['start']
        self.end_token_idx = vocab['end']
        self.unk_token_idx = vocab['unk']
        self.conv_bos_id = vocab['start']
        self.cls_id = vocab['start']
        self.sep_id = vocab['end']
        if 'sent_split' in vocab:
            self.sent_split_idx = vocab['sent_split']
        else:
            self.sent_split_idx = vocab['end']
        if 'word_split' in vocab:
            self.word_split_idx = vocab['word_split']
        else:
            self.word_split_idx = vocab['end']

        self.pad_entity_idx = vocab['pad_entity']
        self.pad_word_idx = vocab['pad_word']
        if 'pad_topic' in vocab:
            self.pad_topic_idx = vocab['pad_topic']

        self.tok2ind = vocab['tok2ind']
        self.ind2tok = vocab['ind2tok']
        self.id2entity = vocab['id2entity']
        if 'ind2topic' in vocab:
            self.ind2topic = vocab['ind2topic']

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
        compact_context = []
        for i, utterance in enumerate(context_tokens):
            if i != 0:
                utterance.insert(0, self.sent_split_idx)
            compact_context.append(utterance)
        compat_context = truncate(merge_utt(compact_context),
                                  self.context_truncate - 2,
                                  truncate_tail=False)
        compat_context = add_start_end_token_idx(compat_context,
                                                 self.start_token_idx,
                                                 self.end_token_idx)
        return compat_context

    def _neg_sample(self, item_set):
        item = random.randint(1, self.item_size)
        while item in item_set:
            item = random.randint(1, self.item_size)
        return item

    def _process_history(self, context_items, item_id=None):
        input_ids = truncate(context_items,
                             max_length=self.item_truncate,
                             truncate_tail=False)
        input_mask = [1] * len(input_ids)
        sample_negs = []
        seq_set = set(input_ids)
        for _ in input_ids:
            sample_negs.append(self._neg_sample(seq_set))

        if item_id is not None:
            target_pos = input_ids[1:] + [item_id]
            return input_ids, target_pos, input_mask, sample_negs
        else:
            return input_ids, input_mask, sample_negs

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

            if 'interaction_history' in conv_dict:
                context_items = conv_dict['interaction_history'] + conv_dict[
                    'context_items']
            else:
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
        batch_mask = (batch_context != self.pad_token_idx).long()

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

    def rec_interact(self, data):
        context = [self._process_rec_context(data['context_tokens'])]
        if 'interaction_history' in data:
            context_items = data['interaction_history'] + data['context_items']
        else:
            context_items = data['context_items']
        input_ids, input_mask, sample_negs = self._process_history(context_items)
        input_ids, input_mask, sample_negs = [input_ids], [input_mask], [sample_negs]

        context = padded_tensor(context,
                                self.pad_token_idx,
                                max_len=self.context_truncate)
        mask = (context != self.pad_token_idx).long()

        return (context, mask,
                padded_tensor(input_ids,
                              pad_idx=self.pad_token_idx,
                              pad_tail=False,
                              max_len=self.item_truncate),
                None,
                padded_tensor(input_mask,
                              pad_idx=self.pad_token_idx,
                              pad_tail=False,
                              max_len=self.item_truncate),
                padded_tensor(sample_negs,
                              pad_idx=self.pad_token_idx,
                              pad_tail=False,
                              max_len=self.item_truncate),
                None)

    def conv_batchify(self, batch):
        batch_context_tokens = []
        batch_enhanced_context_tokens = []
        batch_response = []
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
                    start_token_idx=self.start_token_idx,
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

            enhanced_topic = []
            if 'target' in conv_dict:
                for target_policy in conv_dict['target']:
                    topic_variable = target_policy[1]
                    if isinstance(topic_variable, list):
                        for topic in topic_variable:
                            enhanced_topic.append(topic)
                enhanced_topic = [[
                    self.tok2ind.get(token, self.unk_token_idx) for token in self.ind2topic[topic_id]
                ] for topic_id in enhanced_topic]
                enhanced_topic = merge_utt(enhanced_topic, self.word_split_idx, False, self.sent_split_idx)

            enhanced_movie = []
            if 'items' in conv_dict:
                for movie_id in conv_dict['items']:
                    enhanced_movie.append(movie_id)
                enhanced_movie = [
                    [self.tok2ind.get(token, self.unk_token_idx) for token in self.id2entity[movie_id].split('（')[0]]
                    for movie_id in enhanced_movie]
                enhanced_movie = truncate(merge_utt(enhanced_movie, self.word_split_idx, self.sent_split_idx),
                                          self.item_truncate, truncate_tail=False)

            if len(enhanced_movie) != 0:
                enhanced_context_tokens = enhanced_movie + truncate(batch_context_tokens[-1],
                                                                    max_length=self.context_truncate - len(
                                                                        enhanced_movie), truncate_tail=False)
            elif len(enhanced_topic) != 0:
                enhanced_context_tokens = enhanced_topic + truncate(batch_context_tokens[-1],
                                                                    max_length=self.context_truncate - len(
                                                                        enhanced_topic), truncate_tail=False)
            else:
                enhanced_context_tokens = batch_context_tokens[-1]
            batch_enhanced_context_tokens.append(
                enhanced_context_tokens
            )

        batch_context_tokens = padded_tensor(items=batch_context_tokens,
                                             pad_idx=self.pad_token_idx,
                                             max_len=self.context_truncate,
                                             pad_tail=False)
        batch_response = padded_tensor(batch_response,
                                       pad_idx=self.pad_token_idx,
                                       max_len=self.response_truncate,
                                       pad_tail=True)
        batch_input_ids = torch.cat((batch_context_tokens, batch_response), dim=1)
        batch_enhanced_context_tokens = padded_tensor(items=batch_enhanced_context_tokens,
                                                      pad_idx=self.pad_token_idx,
                                                      max_len=self.context_truncate,
                                                      pad_tail=False)
        batch_enhanced_input_ids = torch.cat((batch_enhanced_context_tokens, batch_response), dim=1)

        return (batch_enhanced_input_ids, batch_enhanced_context_tokens,
                batch_input_ids, batch_context_tokens,
                padded_tensor(batch_context_entities,
                              self.pad_entity_idx,
                              pad_tail=False),
                padded_tensor(batch_context_words,
                              self.pad_word_idx,
                              pad_tail=False), batch_response)

    def conv_interact(self, data):
        context_tokens = [utter + [self.conv_bos_id] for utter in data['context_tokens']]
        context_tokens[-1] = context_tokens[-1][:-1]
        context_tokens = [truncate(merge_utt(context_tokens), max_length=self.context_truncate, truncate_tail=False)]
        context_tokens = padded_tensor(items=context_tokens,
                                       pad_idx=self.pad_token_idx,
                                       max_len=self.context_truncate,
                                       pad_tail=False)
        context_entities = [truncate(data['context_entities'], self.entity_truncate, truncate_tail=False)]
        context_words = [truncate(data['context_words'], self.word_truncate, truncate_tail=False)]

        return (context_tokens, context_tokens,
                context_tokens, context_tokens,
                padded_tensor(context_entities,
                              self.pad_entity_idx,
                              pad_tail=False),
                padded_tensor(context_words,
                              self.pad_word_idx,
                              pad_tail=False), None)

    def policy_process_fn(self, *args, **kwargs):
        augment_dataset = []
        for conv_dict in tqdm(self.dataset):
            for target_policy in conv_dict['target']:
                topic_variable = target_policy[1]
                for topic in topic_variable:
                    augment_conv_dict = deepcopy(conv_dict)
                    augment_conv_dict['target_topic'] = topic
                    augment_dataset.append(augment_conv_dict)
        return augment_dataset

    def policy_batchify(self, batch):
        batch_context = []
        batch_context_policy = []
        batch_user_profile = []
        batch_target = []

        for conv_dict in batch:
            final_topic = conv_dict['final']
            final_topic = [[
                self.tok2ind.get(token, self.unk_token_idx) for token in self.ind2topic[topic_id]
            ] for topic_id in final_topic[1]]
            final_topic = merge_utt(final_topic, self.word_split_idx, False, self.sep_id)

            context = conv_dict['context_tokens']
            context = merge_utt(context,
                                self.sent_split_idx,
                                False,
                                self.sep_id)
            context += final_topic
            context = add_start_end_token_idx(
                truncate(context, max_length=self.context_truncate - 1, truncate_tail=False),
                start_token_idx=self.cls_id)
            batch_context.append(context)

            # [topic, topic, ..., topic]
            context_policy = []
            for policies_one_turn in conv_dict['context_policy']:
                if len(policies_one_turn) != 0:
                    for policy in policies_one_turn:
                        for topic_id in policy[1]:
                            if topic_id != self.pad_topic_idx:
                                policy = []
                                for token in self.ind2topic[topic_id]:
                                    policy.append(self.tok2ind.get(token, self.unk_token_idx))
                                context_policy.append(policy)
            context_policy = merge_utt(context_policy, self.word_split_idx, False)
            context_policy = add_start_end_token_idx(
                context_policy,
                start_token_idx=self.cls_id,
                end_token_idx=self.sep_id)
            context_policy += final_topic
            batch_context_policy.append(context_policy)

            batch_user_profile.extend(conv_dict['user_profile'])

            batch_target.append(conv_dict['target_topic'])

        batch_context = padded_tensor(batch_context,
                                      pad_idx=self.pad_token_idx,
                                      pad_tail=True,
                                      max_len=self.context_truncate)
        batch_cotnext_mask = (batch_context != self.pad_token_idx).long()
        batch_context_policy = padded_tensor(batch_context_policy,
                                             pad_idx=self.pad_token_idx,
                                             pad_tail=True)
        batch_context_policy_mask = (batch_context_policy != 0).long()
        batch_user_profile = padded_tensor(batch_user_profile,
                                           pad_idx=self.pad_token_idx,
                                           pad_tail=True)
        batch_user_profile_mask = (batch_user_profile != 0).long()
        batch_target = torch.tensor(batch_target, dtype=torch.long)

        return (batch_context, batch_cotnext_mask, batch_context_policy,
                batch_context_policy_mask, batch_user_profile,
                batch_user_profile_mask, batch_target)
