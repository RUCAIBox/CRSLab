# @Time   : 2020/12/9
# @Author : Yuanhang Zhou
# @Email  : sdzyh002@gmail.com

# UPDATE:
# @Time   : 2021/1/7, 2021/1/4
# @Author : Xiaolei Wang, Yuanhang Zhou
# @Email  : wxl1999@foxmail.com, sdzyh002@gmail.com

r"""
TGReDial_Rec
============
References:
    Zhou, Kun, et al. `"Towards Topic-Guided Conversational Recommender System."`_ in COLING 2020.

.. _`"Towards Topic-Guided Conversational Recommender System."`:
   https://www.aclweb.org/anthology/2020.coling-main.365/

"""

import os

import torch
from loguru import logger
from torch import nn
from transformers import BertModel

from crslab.config import PRETRAIN_PATH
from crslab.data import dataset_language_map
from crslab.model.base import BaseModel
from crslab.model.pretrained_models import resources
from crslab.model.recommendation.sasrec.modules import SASRec


class TGRecModel(BaseModel):
    """
        
    Attributes:
        hidden_dropout_prob: A float indicating the dropout rate to dropout hidden state in SASRec.
        initializer_range: A float indicating the range of parameters initization in SASRec.
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

        language = dataset_language_map[opt['dataset']]
        resource = resources['bert'][language]
        dpath = os.path.join(PRETRAIN_PATH, "bert", language)
        super(TGRecModel, self).__init__(opt, device, dpath, resource)

    def build_model(self):
        # build BERT layer, give the architecture, load pretrained parameters
        self.bert = BertModel.from_pretrained(self.dpath)
        self.bert_hidden_size = self.bert.config.hidden_size
        self.concat_embed_size = self.bert_hidden_size + self.hidden_size
        self.fusion = nn.Linear(self.concat_embed_size, self.item_size)
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

        bert_embed = self.bert(context, attention_mask=mask).pooler_output

        sequence_output = self.SASREC(input_ids, input_mask)  # bs, max_len, hidden_size2
        sas_embed = sequence_output[:, -1, :]  # bs, hidden_size2

        embed = torch.cat((sas_embed, bert_embed), dim=1)
        rec_scores = self.fusion(embed)  # bs, item_size

        if mode == 'infer':
            return rec_scores
        else:
            rec_loss = self.rec_loss(rec_scores, y)
            return rec_loss, rec_scores
