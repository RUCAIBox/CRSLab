# @Time   : 2020/12/16
# @Author : Yuanhang Zhou
# @Email  : sdzyh002@gmail.com

# UPDATE
# @Time   : 2020/12/29, 2021/1/4
# @Author : Xiaolei Wang, Yuanhang Zhou
# @email  : wxl1999@foxmail.com, sdzyh002@gmail.com

r"""
Popularity
==========
"""

from collections import defaultdict

import torch
from loguru import logger

from crslab.model.base import BaseModel


class PopularityModel(BaseModel):
    """

    Attributes:
        item_size: A integer indicating the number of items.

    """

    def __init__(self, opt, device, vocab, side_data):
        """

        Args:
            opt (dict): A dictionary record the hyper parameters.
            device (torch.device): A variable indicating which device to place the data and model.
            vocab (dict): A dictionary record the vocabulary information.
            side_data (dict): A dictionary record the side data.

        """
        self.item_size = vocab['n_entity']
        super(PopularityModel, self).__init__(opt, device)

    def build_model(self):
        self.item_frequency = defaultdict(int)
        logger.debug('[Finish build rec layer]')

    def forward(self, batch, mode):
        context, mask, input_ids, target_pos, input_mask, sample_negs, y = batch
        if mode == 'train':
            for ids in input_ids:
                for id in ids:
                    self.item_frequency[id.item()] += 1

        bs = input_ids.shape[0]
        rec_score = [self.item_frequency.get(item_id, 0) for item_id in range(self.item_size)]
        rec_scores = torch.tensor([rec_score] * bs, dtype=torch.long)
        loss = torch.zeros(1, requires_grad=True)
        return loss, rec_scores
