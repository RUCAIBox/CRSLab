# @Time   : 2021/3/10
# @Author : Beichen Zhang
# @Email  : zhangbeichen724@gmail.com

import os

import torch
from transformers import GPT2LMHeadModel

from crslab.config import PRETRAIN_PATH
from crslab.data import dataset_language_map
from crslab.model.base import BaseModel
from crslab.model.pretrained_models import resources
from .modules import SequenceCrossEntropyLoss


class InspiredConvModel(BaseModel):
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
        self.label_smoothing = opt['conv']['label_smoothing'] if 'label_smoothing' in opt['conv'] else -1

        language = dataset_language_map[opt['dataset']]
        resource = resources['gpt2'][language]
        dpath = os.path.join(PRETRAIN_PATH, "gpt2", language)
        super(InspiredConvModel, self).__init__(opt, device, dpath, resource)

    def build_model(self):
        """build model for seeker and recommender separately"""
        self.model_sk = GPT2LMHeadModel.from_pretrained(self.dpath)
        self.model_rm = GPT2LMHeadModel.from_pretrained(self.dpath)
        self.loss = SequenceCrossEntropyLoss(self.pad_id, self.label_smoothing)

    def converse(self, batch, mode):
        """
        Args:
            batch: ::

                {
                    'roles': (batch_size),
                    'input_ids': (batch_size, max_seq_length),
                    'context': (batch_size, context_truncate)
                }

        """
        roles, input_ids, context, _ = batch
        input_ids_iters = input_ids.unsqueeze(1)

        past = None
        lm_logits_all = []

        if mode != 'test':
            for turn, iter in enumerate(input_ids_iters):
                if (roles[turn] == 0):
                    # considering that gpt2 only supports up to 1024 tokens
                    if past is not None and past[0].shape[3] + iter.shape[1] > 1024:
                        past = None
                    outputs = self.model_sk(iter, past_key_values=past)
                    lm_logits, past = outputs.logits, outputs.past_key_values
                    lm_logits_all.append(lm_logits)
                else:
                    if past is not None and past[0].shape[3] + iter.shape[1] > 1024:
                        past = None
                    outputs = self.model_rm(iter, past_key_values=past)
                    lm_logits, past = outputs.logits, outputs.past_key_values
                    lm_logits_all.append(lm_logits)

            lm_logits_all = torch.cat(lm_logits_all, dim=0)  # (b_s, seq_len, vocab_size)

            # index from 1 to self.reponse_truncate is valid response
            loss = self.calculate_loss(
                lm_logits_all[:, -self.response_truncate:-1, :],
                input_ids[:, -self.response_truncate + 1:])

            pred = torch.max(lm_logits_all, dim=2)[1]  # (b_s, seq_len)
            pred = pred[:, -self.response_truncate:]

            return loss, pred
        else:
            return self.generate(roles, context)

    def generate(self, roles, context):
        """
        Args:
            roles: the role of each speak corresponding to the utterance in batch, shape=(b_s)
            context: torch.tensor, shape=(b_s, context_turncate)

        Returns:
            generated_response: torch.tensor, shape=(b_s, reponse_turncate-1)
        """
        generated_response = []
        former_hidden_state = None
        context = context[..., -self.response_truncate + 1:]

        for i in range(self.response_truncate - 1):
            last_hidden_state_all = []
            context_iters = context.unsqueeze(1)
            for turn, iter in enumerate(context_iters):
                if roles[turn] == 0:
                    outputs = self.model_sk(iter, former_hidden_state)  # (1, s_l, v_s),
                else:
                    outputs = self.model_rm(iter, former_hidden_state)  # (1, s_l, v_s),
                last_hidden_state, former_hidden_state = outputs.logits, outputs.past_key_values
                last_hidden_state_all.append(last_hidden_state)

            last_hidden_state_all = torch.cat(last_hidden_state_all, dim=0)
            next_token_logits = last_hidden_state_all[:, -1, :]  # (b_s, v_s)
            preds = next_token_logits.argmax(dim=-1).long()  # (b_s)

            context = preds.unsqueeze(1)
            generated_response.append(preds)

        generated_response = torch.stack(generated_response).T

        return generated_response

    def calculate_loss(self, logit, labels):
        """

        Args:
            preds: torch.FloatTensor, shape=(b_s, response_truncate, vocab_size)
            labels: torch.LongTensor, shape=(b_s, response_truncate)

        """

        loss = self.loss(logit, labels)
        return loss
