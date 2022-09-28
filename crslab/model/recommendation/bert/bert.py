# @Time   : 2020/12/16
# @Author : Yuanhang Zhou
# @Email  : sdzyh002@gmail.com

# UPDATE
# @Time   : 2021/1/7, 2021/1/4
# @Author : Xiaolei Wang, Yuanhang Zhou
# @email  : wxl1999@foxmail.com, sdzyh002@gmail.com

# UPDATE:
# @Time   : 2022/9/28
# @Author : Xinyu Tang
# @Email  : txy20010310@163.com

r"""
BERT
====
References:
    Devlin, Jacob, et al. `"BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding."`_ in NAACL 2019.

.. _`"BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding."`:
   https://www.aclweb.org/anthology/N19-1423/

"""

import os

from loguru import logger
from torch import nn
from transformers import BertModel

from crslab.config import PRETRAIN_PATH, BERT_EN_PATH, BERT_ZH_PATH
from crslab.data import dataset_language_map
from crslab.model.base import BaseModel


class BERTModel(BaseModel):
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

        self.language = dataset_language_map[opt['dataset']]
        self.dpath = os.path.join(PRETRAIN_PATH, "bert", self.language)
        super(BERTModel, self).__init__(opt, device, self.dpath)

    def build_model(self):
        # build BERT layer, give the architecture, load pretrained parameters
        if os.path.exists(self.dpath):
            self.bert = BertModel.from_pretrained(self.dpath)
        else:
            os.makedirs(self.dpath)
            if self.language == 'zh':
                os.environ['TORCH_HOME'] = BERT_ZH_PATH
                self.bert = BertModel.from_pretrained('base-base-chinese')
            elif self.language == 'en':
                os.environ['TORCH_HOME'] = BERT_EN_PATH
                self.bert = BertModel.from_pretrained('bert-base-uncased')
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
