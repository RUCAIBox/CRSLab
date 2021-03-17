# @Time   : 2020/12/16
# @Author : Yuanhang Zhou
# @Email  : sdzyh002@gmail.com

# UPDATE
# @Time   : 2020/12/29, 2021/1/4
# @Author : Xiaolei Wang, Yuanhang Zhou
# @email  : wxl1999@foxmail.com, sdzyh002@gmail.com

r"""
TextCNN
=======
References:
    Kim, Yoon. `"Convolutional Neural Networks for Sentence Classification."`_ in EMNLP 2014.

.. _`"Convolutional Neural Networks for Sentence Classification."`:
   https://www.aclweb.org/anthology/D14-1181/

"""

import torch
import torch.nn.functional as F
from loguru import logger
from torch import nn

from crslab.model.base import BaseModel


class TextCNNModel(BaseModel):
    """
        
    Attributes:
        movie_num: A integer indicating the number of items.
        num_filters: A string indicating the number of filter in CNN.
        embed: A integer indicating the size of embedding layer.
        filter_sizes: A string indicating the size of filter in CNN.
        dropout: A float indicating the dropout rate.

    """

    def __init__(self, opt, device, vocab, side_data):
        """

        Args:
            opt (dict): A dictionary record the hyper parameters.
            device (torch.device): A variable indicating which device to place the data and model.
            vocab (dict): A dictionary record the vocabulary information.
            side_data (dict): A dictionary record the side data.

        """
        self.movie_num = vocab['n_entity']
        self.num_filters = opt['num_filters']
        self.embed = opt['embed']
        self.filter_sizes = eval(opt['filter_sizes'])
        self.dropout = opt['dropout']
        super(TextCNNModel, self).__init__(opt, device)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def build_model(self):
        self.embedding = nn.Embedding(self.movie_num, self.embed)

        self.convs = nn.ModuleList(
            [nn.Conv2d(1, self.num_filters, (k, self.embed)) for k in self.filter_sizes])
        self.dropout = nn.Dropout(self.dropout)
        self.fc = nn.Linear(self.num_filters * len(self.filter_sizes), self.movie_num)

        # this loss may conduct to some weakness
        self.rec_loss = nn.CrossEntropyLoss()

        logger.debug('[Finish build rec layer]')

    def forward(self, batch, mode):
        context, mask, input_ids, target_pos, input_mask, sample_negs, y = batch

        out = self.embedding(context)
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)

        rec_scores = out
        rec_loss = self.rec_loss(out, y)

        return rec_loss, rec_scores
