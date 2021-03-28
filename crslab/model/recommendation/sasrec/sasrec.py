# @Time   : 2020/12/16
# @Author : Yuanhang Zhou
# @Email  : sdzyh002@gmail.com

# UPDATE
# @Time   : 2020/12/29, 2021/1/4
# @Author : Xiaolei Wang, Yuanhang Zhou
# @email  : wxl1999@foxmail.com, sdzyh002@gmail.com

r"""
SASREC
======
References:
    Kang, Wang-Cheng, and Julian McAuley. `"Self-attentive sequential recommendation."`_ in ICDM 2018.

.. _`"Self-attentive sequential recommendation."`:
   https://ieeexplore.ieee.org/abstract/document/8594844

"""

import torch
from loguru import logger
from torch import nn

from crslab.model.base import BaseModel
from crslab.model.recommendation.sasrec.modules import SASRec


class SASRECModel(BaseModel):
    """
        
    Attributes:
        hidden_dropout_prob: A float indicating the dropout rate to dropout hidden state in SASRec.
        initializer_range: A float indicating the range of parameters initiation in SASRec.
        hidden_size: A integer indicating the size of hidden state in SASRec.
        max_seq_length: A integer indicating the max interaction history length.
        item_size: A integer indicating the number of items.
        num_attention_heads: A integer indicating the head number in SASRec.
        attention_probs_dropout_prob: A float indicating the dropout rate in attention layers.
        hidden_act: A string indicating the activation function type in SASRec.
        num_hidden_layers: A integer indicating the number of hidden layers in SASRec.

    """

    def __init__(self, opt, device, vocab, side_data):
        """

        Args:
            opt (dict): A dictionary record the hyper parameters.
            device (torch.device): A variable indicating which device to place the data and model.
            vocab (dict): A dictionary record the vocabulary information.
            side_data (dict): A dictionary record the side data.

        """
        self.hidden_dropout_prob = opt['hidden_dropout_prob']
        self.initializer_range = opt['initializer_range']
        self.hidden_size = opt['hidden_size']
        self.max_seq_length = opt['max_history_items']
        self.item_size = vocab['n_entity'] + 1
        self.num_attention_heads = opt['num_attention_heads']
        self.attention_probs_dropout_prob = opt['attention_probs_dropout_prob']
        self.hidden_act = opt['hidden_act']
        self.num_hidden_layers = opt['num_hidden_layers']

        super(SASRECModel, self).__init__(opt, device)

    def build_model(self):
        # build BERT layer, give the architecture, load pretrained parameters
        self.SASREC = SASRec(self.hidden_dropout_prob, self.device,
                             self.initializer_range, self.hidden_size,
                             self.max_seq_length, self.item_size,
                             self.num_attention_heads,
                             self.attention_probs_dropout_prob,
                             self.hidden_act, self.num_hidden_layers)

        # this loss may conduct to some weakness
        self.rec_loss = nn.CrossEntropyLoss()

        logger.debug('[Finish build rec layer]')

    def forward(self, batch, mode):
        context, mask, input_ids, target_pos, input_mask, sample_negs, y = batch
        # print(input_ids.shape)
        sequence_output = self.SASREC(input_ids, input_mask)  # bs, max_len, hidden_size2

        logit = sequence_output[:, -1:, :]
        rec_scores = torch.matmul(logit, self.SASREC.embeddings.item_embeddings.weight.data.T)
        rec_scores = rec_scores.squeeze(1)
        # print('rec_scores.shape', rec_scores.shape)

        rec_loss = self.SASREC.cross_entropy(sequence_output, target_pos,
                                             sample_negs)

        return rec_loss, rec_scores
