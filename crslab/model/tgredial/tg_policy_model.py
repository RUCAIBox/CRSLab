# @Time   : 2020/12/9
# @Author : Yuanhang Zhou
# @Email  : sdzyh002@gmail.com

# UPDATE:
# @Time   : 2020/12/13
# @Author : Xiaolei Wang
# @Email  : wxl1999@foxmail.com

import os

import torch
from torch import nn
from transformers import BertModel

from crslab.model.base_model import BaseModel
from .resource import resources
from ...config.config import MODEL_PATH


class TGPolicyModel(BaseModel):
    def __init__(self, opt, device, vocab, side_data):
        self.topic_class_num = vocab['n_topic']
        dataset = opt['dataset']
        dpath = os.path.join(MODEL_PATH, "tgredial", dataset)
        resource = resources[dataset]
        super(TGPolicyModel, self).__init__(opt, device, dpath, resource)

    def build_model(self, *args, **kwargs):
        """build model"""
        self.context_bert = BertModel.from_pretrained(os.path.join(self.dpath, 'Bert'))
        self.topic_bert = BertModel.from_pretrained(os.path.join(self.dpath, 'Bert'))
        self.profile_bert = BertModel.from_pretrained(os.path.join(self.dpath, 'Bert'))

        self.bert_hidden_size = self.context_bert.config.hidden_size
        self.state2topic_id = nn.Linear(self.bert_hidden_size * 3,
                                        self.topic_class_num)

        self.loss = nn.CrossEntropyLoss()

    def forward(self, batch, mode):
        conv_id, message_id, context, context_mask, topic_path_kw, tp_mask, user_profile, profile_mask, y = batch

        context_last_hidden_state, context_rep = self.context_bert(
            context,
            context_mask)  # [bs, seq_len, hidden_size]， [bs, hidden_size]

        tp_last_hidden_state, topic_rep = self.topic_bert(
            topic_path_kw,
            tp_mask)  # [bs, hidden_size], topic_rep = (bs, hiddensize)

        bs, sent_num, word_num = user_profile.shape
        user_profile = user_profile.view(-1, user_profile.shape[-1])
        profile_mask = profile_mask.view(-1, profile_mask.shape[-1])

        profile_last_hidden_state, profile_rep = self.profile_bert(
            user_profile, profile_mask)  # (bs, word_num, hidden)

        profile_rep = profile_rep.view(bs, sent_num, -1)

        profile_rep = torch.mean(profile_rep, dim=1)  # (bs, hidden)

        state_rep = torch.cat((context_rep, topic_rep, profile_rep),
                              1)  # [bs, hidden_size*3]

        topic_scores = self.state2topic_id(state_rep)

        topic_loss = self.loss(topic_scores, y)

        return topic_loss, topic_scores
