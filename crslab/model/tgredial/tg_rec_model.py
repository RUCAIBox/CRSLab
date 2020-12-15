# @Time   : 2020/12/9
# @Author : Yuanhang Zhou
# @Email  : sdzyh002@gmail.com

# UPDATE:
# @Time   : 2020/12/14
# @Author : Xiaolei Wang
# @Email  : wxl1999@foxmail.com

import os

import torch
from loguru import logger
from torch import nn
from transformers import BertModel

from crslab.config.config import MODEL_PATH
from crslab.model.base_model import BaseModel
from .resource import resources
from .sasrec_model import SASRecModel


class TGRecModel(BaseModel):
    def __init__(self, opt, device, vocab, side_data):
        self.hidden_dropout_prob = opt['hidden_dropout_prob']
        self.initializer_range = opt['initializer_range']
        self.hidden_size = opt['hidden_size']
        self.max_seq_length = opt['max_history_items']
        self.item_size = vocab['n_entity']
        self.num_attention_heads = opt['num_attention_heads']
        self.attention_probs_dropout_prob = opt['attention_probs_dropout_prob']
        self.hidden_act = opt['hidden_act']
        self.num_hidden_layers = opt['num_hidden_layers']

        dataset = opt['dataset']
        dpath = os.path.join(MODEL_PATH, "tgredial", dataset)
        resource = resources[dataset]
        super(TGRecModel, self).__init__(opt, device, dpath, resource)

    def build_model(self):
        # build BERT layer, give the architecture, load pretrained parameters
        self.bert = BertModel.from_pretrained(os.path.join(self.dpath, 'bert'))
        self.bert_hidden_size = self.bert.config.hidden_size
        self.concat_embed_size = self.bert_hidden_size + self.hidden_size
        self.fusion = nn.Linear(self.concat_embed_size, self.item_size)
        self.SASREC = SASRecModel(self.hidden_dropout_prob, self.device,
                                  self.initializer_range, self.hidden_size,
                                  self.max_seq_length, self.item_size,
                                  self.num_attention_heads,
                                  self.attention_probs_dropout_prob,
                                  self.hidden_act, self.num_hidden_layers)

        # this loss may conduct to some weakness
        self.rec_loss = nn.CrossEntropyLoss()

        logger.debug('[Finish build rec layer]')

    def forward(self, batch, mode='train'):
        context, mask, input_ids, target_pos, input_mask, sample_negs, y = batch

        bert_embed = self.bert(context, attention_mask=mask).pooler_output

        sequence_output = self.SASREC(input_ids, input_mask)  # bs, max_len, hidden_size2
        sas_embed = sequence_output[:, -1, :]  # bs, hidden_size2

        embed = torch.cat((sas_embed, bert_embed), dim=1)
        rec_scores = self.fusion(embed)  # bs, item_size

        rec_loss = self.rec_loss(rec_scores, y)

        return rec_loss, rec_scores
