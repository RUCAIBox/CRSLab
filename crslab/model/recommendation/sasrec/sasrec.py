# @Time   : 2020/12/16
# @Author : Yuanhang Zhou
# @Email  : sdzyh002@gmail.com

# UPDATE
# @Time   : 2020/12/24
# @Author : Xiaolei Wang
# @email  : wxl1999@foxmail.com

import torch
from loguru import logger
from torch import nn

from crslab.model.base_model import BaseModel
from crslab.model.sasrec_model import SASRecModel


class SASRECModel(BaseModel):
    def __init__(self, opt, device, vocab, side_data):
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
        # print(input_ids.shape)
        sequence_output = self.SASREC(input_ids, input_mask)  # bs, max_len, hidden_size2

        logit = sequence_output[:, -1:, :]
        rec_scores = torch.matmul(logit, self.SASREC.embeddings.item_embeddings.weight.data.T)
        rec_scores = rec_scores.squeeze(1)
        # print('rec_scores.shape', rec_scores.shape)

        rec_loss = self.SASREC.cross_entropy(sequence_output, target_pos,
                                             sample_negs)

        return rec_loss, rec_scores
