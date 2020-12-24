# @Time   : 2020/12/16
# @Author : Yuanhang Zhou
# @Email  : sdzyh002@gmail.com

# UPDATE
# @Time   : 2020/12/24
# @Author : Xiaolei Wang
# @email  : wxl1999@foxmail.com

import os

from loguru import logger
from torch import nn
from transformers import BertModel

from crslab.config import dataset_language_map, MODEL_PATH
from crslab.model.base_model import BaseModel
from .resource import resources


class BERTModel(BaseModel):
    def __init__(self, opt, device, vocab, side_data):
        self.item_size = vocab['n_entity']

        language = dataset_language_map[opt['dataset']]
        dpath = os.path.join(MODEL_PATH, "tgredial", language)
        resource = resources[language]
        super(BERTModel, self).__init__(opt, device, dpath, resource)

    def build_model(self):
        # build BERT layer, give the architecture, load pretrained parameters
        self.bert = BertModel.from_pretrained(os.path.join(self.dpath, 'bert'))
        # print(self.item_size)
        self.bert_hidden_size = self.bert.config.hidden_size
        self.mlp = nn.Linear(self.bert_hidden_size, self.item_size)

        # this loss may conduct to some weakness
        self.rec_loss = nn.CrossEntropyLoss()

        logger.debug('[Finish build rec layer]')

    def forward(self, batch, mode='train'):
        context, mask, input_ids, target_pos, input_mask, sample_negs, y = batch

        bert_embed = self.bert(context, attention_mask=mask).pooler_output

        rec_scores = self.mlp(bert_embed)  # bs, item_size

        rec_loss = self.rec_loss(rec_scores, y)

        return rec_loss, rec_scores
