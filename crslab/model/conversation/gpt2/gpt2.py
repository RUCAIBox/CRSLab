# @Time   : 2020/12/14
# @Author : Yuanhang Zhou
# @Email  : sdzyh002@gmail.com

# UPDATE
# @Time   : 2021/1/7
# @Author : Xiaolei Wang
# @email  : wxl1999@foxmail.com

r"""
GPT2
====
References:
    Radford, Alec, et al. `"Language Models are Unsupervised Multitask Learners."`_.

.. _`"Language Models are Unsupervised Multitask Learners."`:
   https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf

"""

import os

import torch
from torch.nn import CrossEntropyLoss
from transformers import GPT2LMHeadModel

from crslab.config import PRETRAIN_PATH
from crslab.data import dataset_language_map
from crslab.model.base import BaseModel
from crslab.model.pretrained_models import resources


class GPT2Model(BaseModel):
    """
        
    Attributes:
        context_truncate: A integer indicating the length of dialogue context.
        response_truncate: A integer indicating the length of dialogue response.
        pad_id: A integer indicating the id of padding token.

    """

    def __init__(self, opt, device, vocab, side_data):
        """

        Args:
            opt (dict): A dictionary record the hyper parameters.
            device (torch.device): A variable indicating which device to place the data and model.
            vocab (dict): A dictionary record the vocabulary information.
            side_data (dict): A dictionary record the side data.

        """
        self.context_truncate = opt['context_truncate']
        self.response_truncate = opt['response_truncate']
        self.pad_id = vocab['pad']

        language = dataset_language_map[opt['dataset']]
        resource = resources['gpt2'][language]
        dpath = os.path.join(PRETRAIN_PATH, "gpt2", language)
        super(GPT2Model, self).__init__(opt, device, dpath, resource)

    def build_model(self):
        """build model"""
        self.model = GPT2LMHeadModel.from_pretrained(self.dpath)
        self.loss = CrossEntropyLoss(ignore_index=self.pad_id)

    def converse(self, batch, mode):
        _, _, input_ids, context, _, _, y = batch

        if mode != 'test':
            # torch.tensor's shape = (bs, seq_len, v_s); tuple's length = 12
            lm_logits = self.model(input_ids).logits

            # index from 1 to self.reponse_truncate is valid response
            loss = self.calculate_loss(
                lm_logits[:, -self.response_truncate:-1, :],
                input_ids[:, -self.response_truncate + 1:])

            pred = torch.max(lm_logits, dim=2)[1]  # [bs, seq_len]
            pred = pred[:, -self.response_truncate:]

            return loss, pred
        else:
            return self.generate(context)

    def generate(self, context):
        """
        Args:
            context: torch.tensor, shape=(bs, context_turncate)

        Returns:
            generated_response: torch.tensor, shape=(bs, reponse_turncate-1)
        """
        generated_response = []
        former_hidden_state = None
        context = context[..., -self.response_truncate + 1:]

        for i in range(self.response_truncate - 1):
            outputs = self.model(context, former_hidden_state)  # (bs, c_t, v_s),
            last_hidden_state, former_hidden_state = outputs.logits, outputs.past_key_values

            next_token_logits = last_hidden_state[:, -1, :]  # (bs, v_s)
            preds = next_token_logits.argmax(dim=-1).long()  # (bs)

            context = preds.unsqueeze(1)
            generated_response.append(preds)

        generated_response = torch.stack(generated_response).T

        return generated_response

    def calculate_loss(self, logit, labels):
        """
        Args:
            preds: torch.FloatTensor, shape=(bs, response_truncate, vocab_size)
            labels: torch.LongTensor, shape=(bs, response_truncate)

        """

        loss = self.loss(logit.reshape(-1, logit.size(-1)), labels.reshape(-1))
        return loss
