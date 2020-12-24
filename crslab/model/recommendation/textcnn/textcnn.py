# @Time   : 2020/12/16
# @Author : Yuanhang Zhou
# @Email  : sdzyh002@gmail.com

# UPDATE
# @Time   : 2020/12/24
# @Author : Xiaolei Wang
# @email  : wxl1999@foxmail.com

import torch
import torch.nn.functional as F
from loguru import logger
from torch import nn

from crslab.model.base_model import BaseModel


class TextCNNModel(BaseModel):
    def __init__(self, opt, device, vocab, side_data):
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

    def forward(self, batch, mode='train'):
        context, mask, input_ids, target_pos, input_mask, sample_negs, y = batch

        out = self.embedding(context)
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)

        rec_scores = out
        rec_loss = self.rec_loss(out, y)

        return rec_loss, rec_scores
