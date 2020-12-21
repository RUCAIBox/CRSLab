# @Time   : 2020/11/27
# @Author : Xiaolei Wang
# @Email  : wxl1999@foxmail.com

# UPDATE:
# @Time   : 2020/12/2
# @Author : Xiaolei Wang
# @Email  : wxl1999@foxmail.com

import torch
from tqdm import tqdm

from crslab.data.dataloader.base_dataloader import BaseDataLoader
from crslab.data.dataloader.utils import add_start_end_token_idx, padded_tensor, truncate, merge_utt


class KBRDDataLoader(BaseDataLoader):
    def __init__(self, opt, dataset, vocab):
        super().__init__(opt, dataset)
        self.pad_token_idx = vocab['pad']
        self.start_token_idx = vocab['start']
        self.end_token_idx = vocab['end']
        self.pad_entity_idx = vocab['pad_entity']
        self.context_truncate = opt.get('context_truncate', None)
        self.response_truncate = opt.get('response_truncate', None)
        self.entity_truncate = opt.get('entity_truncate', None)

    def rec_process_fn(self):
        """
        Sometimes, the recommender may recommend more than one movies to seeker,
        hence we need to augment data for each recommended movie
        """
        augment_dataset = []
        for conv_dict in tqdm(self.dataset):
            if conv_dict['role'] == 'Recommender':
                for movie in conv_dict['items']:
                    augment_conv_dict = {'context_entities': conv_dict['context_entities'], 'item': movie}
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
        output: torch.tensors (context_entities, movie)
        """
        batch_context_entities = []
        batch_movies = []
        for conv_dict in batch:
            batch_context_entities.append(conv_dict['context_entities'])
            batch_movies.append(conv_dict['item'])

        return {
            "context_entities": batch_context_entities,
            "item": torch.tensor(batch_movies, dtype=torch.long)
        }

    def conv_process_fn(self, *args, **kwargs):
        return self.retain_recommender_target()

    def conv_batchify(self, batch):
        """
        input: conv_dict = {
                    "context_tokens": [id1, id2, ..., ],  # [int]
                    "context_entities": [id1, id2, ..., ],  # [int]
                    "context_words": [id1, id2, ..., ],  # [int]
                    "response": [id1, id2, ..., ],  # [int]
                    "items": [id1, id2, ..., ],  # [int]
                }
        output: torch.tensors (context_tokens, context_entities, response)
        """
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

    def policy_batchify(self, *args, **kwargs):
        pass
