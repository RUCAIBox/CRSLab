# @Time   : 2021/3/11
# @Author : Beichen Zhang
# @Email  : zhangbeichen724@gmail.com

from copy import deepcopy

import torch
from tqdm import tqdm

from crslab.data.dataloader.base import BaseDataLoader
from crslab.data.dataloader.utils import add_start_end_token_idx, padded_tensor, truncate, merge_utt


class InspiredDataLoader(BaseDataLoader):
    """Dataloader for model Inspired.

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

        self.pad_entity_idx = vocab['pad_entity']
        self.pad_word_idx = vocab['pad_word']

        self.tok2ind = vocab['tok2ind']
        self.ind2tok = vocab['ind2tok']
        self.id2entity = vocab['id2entity']

        self.context_truncate = opt.get('context_truncate', None)
        self.response_truncate = opt.get('response_truncate', None)

    def rec_process_fn(self, *args, **kwargs):
        augment_dataset = []
        for conv_dict in tqdm(self.dataset):
            if conv_dict['role'] == 'Recommender':
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

    def rec_batchify(self, batch):
        batch_context = []
        batch_movie_id = []

        for conv_dict in batch:
            context = self._process_rec_context(conv_dict['context_tokens'])
            batch_context.append(context)

            item_id = conv_dict['item']
            batch_movie_id.append(item_id)

        batch_context = padded_tensor(batch_context,
                                      self.pad_token_idx,
                                      max_len=self.context_truncate)
        batch_mask = (batch_context != self.pad_token_idx).long()

        return (batch_context, batch_mask, torch.tensor(batch_movie_id))

    def conv_batchify(self, batch):
        """get batch and corresponding roles
        """
        batch_roles = []
        batch_context_tokens = []
        batch_response = []

        for conv_dict in batch:
            batch_roles.append(0 if conv_dict['role'] == 'Seeker' else 1)
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

        batch_context_tokens = padded_tensor(items=batch_context_tokens,
                                             pad_idx=self.pad_token_idx,
                                             max_len=self.context_truncate,
                                             pad_tail=False)
        batch_response = padded_tensor(batch_response,
                                       pad_idx=self.pad_token_idx,
                                       max_len=self.response_truncate,
                                       pad_tail=True)
        batch_input_ids = torch.cat((batch_context_tokens, batch_response), dim=1)
        batch_roles = torch.tensor(batch_roles)

        return (batch_roles,
                batch_input_ids,
                batch_context_tokens,
                batch_response)

    def policy_batchify(self, batch):
        pass
