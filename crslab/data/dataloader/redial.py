# @Time   : 2020/11/22
# @Author : Chenzhan Shang
# @Email  : czshang@outlook.com

# UPDATE:
# @Time   : 2020/12/16
# @Author : Xiaolei Wang
# @Email  : wxl1999@foxmail.com

import re
from copy import copy

import torch
from tqdm import tqdm

from crslab.data.dataloader.base import BaseDataLoader
from crslab.data.dataloader.utils import padded_tensor, get_onehot, truncate

movie_pattern = re.compile(r'^@\d{5,6}$')


class ReDialDataLoader(BaseDataLoader):
    """Dataloader for model ReDial.

    Notes:
        You can set the following parameters in config:

        - ``'utterance_truncate'``: the maximum length of a single utterance.
        - ``'conversation_truncate'``: the maximum length of the whole conversation.

        The following values must be specified in ``vocab``:

        - ``'pad'``
        - ``'start'``
        - ``'end'``
        - ``'unk'``

        the above values specify the id of needed special token.

        - ``'ind2tok'``: map from index to token.
        - ``'n_entity'``: number of entities in the entity KG of dataset.
        - ``'vocab_size'``: size of vocab.

    """

    def __init__(self, opt, dataset, vocab):
        """

        Args:
            opt (Config or dict): config for dataloader or the whole system.
            dataset: data for model.
            vocab (dict): all kinds of useful size, idx and map between token and idx.

        """
        super().__init__(opt, dataset)
        self.ind2tok = vocab['ind2tok']
        self.n_entity = vocab['n_entity']
        self.pad_token_idx = vocab['pad']
        self.start_token_idx = vocab['start']
        self.end_token_idx = vocab['end']
        self.unk_token_idx = vocab['unk']
        self.item_token_idx = vocab['vocab_size']
        self.conversation_truncate = self.opt.get('conversation_truncate', None)
        self.utterance_truncate = self.opt.get('utterance_truncate', None)

    def rec_process_fn(self, *args, **kwargs):
        dataset = []
        for conversation in self.dataset:
            if conversation['role'] == 'Recommender':
                for item in conversation['items']:
                    context_entities = conversation['context_entities']
                    dataset.append({'context_entities': context_entities, 'item': item})
        return dataset

    def rec_batchify(self, batch):
        batch_context_entities = []
        batch_item = []
        for conversation in batch:
            batch_context_entities.append(conversation['context_entities'])
            batch_item.append(conversation['item'])
        context_entities = get_onehot(batch_context_entities, self.n_entity)
        return {'context_entities': context_entities, 'item': torch.tensor(batch_item, dtype=torch.long)}

    def conv_process_fn(self):
        dataset = []
        for conversation in tqdm(self.dataset):
            if conversation['role'] != 'Recommender':
                continue
            context_tokens = [truncate(utterance, self.utterance_truncate, truncate_tail=True) for utterance in
                              conversation['context_tokens']]
            context_tokens = truncate(context_tokens, self.conversation_truncate, truncate_tail=True)
            context_length = len(context_tokens)
            utterance_lengths = [len(utterance) for utterance in context_tokens]
            request = context_tokens[-1]
            response = truncate(conversation['response'], self.utterance_truncate, truncate_tail=True)
            dataset.append({'context_tokens': context_tokens, 'context_length': context_length,
                            'utterance_lengths': utterance_lengths, 'request': request, 'response': response})
        return dataset

    def conv_batchify(self, batch):
        max_utterance_length = max([max(conversation['utterance_lengths']) for conversation in batch])
        max_response_length = max([len(conversation['response']) for conversation in batch])
        max_utterance_length = max(max_utterance_length, max_response_length)
        max_context_length = max([conversation['context_length'] for conversation in batch])
        batch_context = []
        batch_context_length = []
        batch_utterance_lengths = []
        batch_request = []  # tensor
        batch_request_length = []
        batch_response = []

        for conversation in batch:
            padded_context = padded_tensor(conversation['context_tokens'], pad_idx=self.pad_token_idx,
                                           pad_tail=True, max_len=max_utterance_length)
            if len(conversation['context_tokens']) < max_context_length:
                pad_tensor = padded_context.new_full(
                    (max_context_length - len(conversation['context_tokens']), max_utterance_length), self.pad_token_idx
                )
                padded_context = torch.cat((padded_context, pad_tensor), 0)
            batch_context.append(padded_context)
            batch_context_length.append(conversation['context_length'])
            batch_utterance_lengths.append(conversation['utterance_lengths'] +
                                           [0] * (max_context_length - len(conversation['context_tokens'])))

            request = conversation['request']
            batch_request_length.append(len(request))
            batch_request.append(request)

            response = copy(conversation['response'])
            # replace '^\d{5,6}$' by '__item__'
            for i in range(len(response)):
                if movie_pattern.match(self.ind2tok[response[i]]):
                    response[i] = self.item_token_idx
            batch_response.append(response)

        context = torch.stack(batch_context, dim=0)
        request = padded_tensor(batch_request, self.pad_token_idx, pad_tail=True, max_len=max_utterance_length)
        response = padded_tensor(batch_response, self.pad_token_idx, pad_tail=True,
                                 max_len=max_utterance_length)  # (bs, utt_len)

        return {'context': context, 'context_lengths': torch.tensor(batch_context_length),
                'utterance_lengths': torch.tensor(batch_utterance_lengths), 'request': request,
                'request_lengths': torch.tensor(batch_request_length), 'response': response}

    def policy_batchify(self, batch):
        pass
