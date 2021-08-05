# @Time   : 2020/11/27
# @Author : Xiaolei Wang
# @Email  : wxl1999@foxmail.com

# UPDATE:
# @Time   : 2020/12/2, 2021/8/5
# @Author : Xiaolei Wang, Chenzhan Shang
# @Email  : wxl1999@foxmail.com, czshang@outlook.com

import torch
from tqdm import tqdm

from crslab.agent.supervised.base import SupervisedAgent
from crslab.agent.supervised.utils import add_start_end_token_idx, padded_tensor, truncate, merge_utt


class KBRDAgent(SupervisedAgent):
    """Agent for model KBRD.

    Notes:
        You can set the following parameters in config:

        - ``'context_truncate'``: the maximum length of context.
        - ``'response_truncate'``: the maximum length of response.
        - ``'entity_truncate'``: the maximum length of mentioned entities in context.

        The following values must be specified in ``vocab``:

        - ``'pad'``
        - ``'start'``
        - ``'end'``
        - ``'pad_entity'``

        the above values specify the id of needed special token.

    """

    def __init__(self, opt, dataset):
        """

        Args:
            opt (Config or dict): config for supervised or the whole system.
            dataset: data for model.

        """
        super().__init__(opt, dataset)
        self.pad_token_idx = dataset.vocab['pad']
        self.start_token_idx = dataset.vocab['start']
        self.end_token_idx = dataset.vocab['end']
        self.pad_entity_idx = dataset.vocab['pad_entity']
        self.context_truncate = opt.get('context_truncate', None)
        self.response_truncate = opt.get('response_truncate', None)
        self.entity_truncate = opt.get('entity_truncate', None)

    def rec_process_fn(self, mode):
        augment_dataset = []
        for conv_dict in tqdm(self.data[mode]):
            if conv_dict['role'] == 'Recommender':
                for movie in conv_dict['items']:
                    augment_conv_dict = {'context_entities': conv_dict['context_entities'], 'item': movie}
                    augment_dataset.append(augment_conv_dict)
        return augment_dataset

    def rec_batchify(self, batch):
        batch_context_entities = []
        batch_movies = []
        for conv_dict in batch:
            batch_context_entities.append(conv_dict['context_entities'])
            batch_movies.append(conv_dict['item'])

        return {
            "context_entities": batch_context_entities,
            "item": torch.tensor(batch_movies, dtype=torch.long)
        }

    def conv_process_fn(self, mode):
        return self.retain_recommender_target(mode)

    def conv_batchify(self, batch):
        batch_context_tokens = []
        batch_context_entities = []
        batch_response = []
        for conv_dict in batch:
            batch_context_tokens.append(
                truncate(merge_utt(conv_dict['context_tokens']), self.context_truncate, truncate_tail=False))
            batch_context_entities.append(conv_dict['context_entities'])
            batch_response.append(
                add_start_end_token_idx(truncate(conv_dict['response'], self.response_truncate - 2),
                                        start_token_idx=self.start_token_idx,
                                        end_token_idx=self.end_token_idx))

        return {
            "context_tokens": padded_tensor(batch_context_tokens, self.pad_token_idx, pad_tail=False),
            "context_entities": batch_context_entities,
            "response": padded_tensor(batch_response, self.pad_token_idx)
        }
