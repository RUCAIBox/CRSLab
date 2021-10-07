# @Time   : 2021/3/10
# @Author : Beichen Zhang
# @Email  : zhangbeichen724@gmail.com

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from transformers import GPT2LMHeadModel, BertModel

from crslab.model.base import BaseModel
from crslab.model.pretrained_models import resources


class SequenceCrossEntropyLoss(nn.Module):
    """

    Attributes:
        ignore_index: indices corresponding tokens which should be ignored in calculating loss.
        label_smoothing: determine smoothing value in cross entropy loss. should be less than 1.0.

    """

    def __init__(self, ignore_index=None, label_smoothing=-1):
        super().__init__()
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(self, logits, labels):
        """

        Args:
            logits: (batch_size, max_seq_len, vocal_size)
            labels: (batch_size, max_seq_len)

        """
        if self.label_smoothing > 1.0:
            raise ValueError('The param label_smoothing should be in the range of 0.0 to 1.0.')
        if self.ignore_index == None:
            mask = torch.ones_like(labels, dtype=torch.float)
        else:
            mask = (labels != self.ignore_index).float()
        logits_flat = logits.reshape(-1, logits.size(-1))  # (b_s * s_l, num_classes)
        log_probs_flat = F.log_softmax(logits_flat, dim=-1)
        labels_flat = labels.reshape(-1, 1).long()  # (b_s * s_l, 1)

        if self.label_smoothing > 0.0:
            num_classes = logits.size(-1)
            smoothing_value = self.label_smoothing / float(num_classes)
            one_hot_labels = torch.zeros_like(log_probs_flat).scatter_(-1, labels_flat,
                                                                       1.0 - self.label_smoothing)  # fill all the correct indices with 1 - smoothing value.
            smoothed_labels = one_hot_labels + smoothing_value
            negative_log_likelihood_flat = -log_probs_flat * smoothed_labels
            negative_log_likelihood_flat = negative_log_likelihood_flat.sum(-1, keepdim=True)
        else:
            negative_log_likelihood_flat = -torch.gather(log_probs_flat, dim=1, index=labels_flat)  # (b_s * s_l, 1)

        negative_log_likelihood = negative_log_likelihood_flat.view(-1, logits.shape[1])  # (b_s, s_l)
        loss = negative_log_likelihood * mask

        loss = loss.sum(1) / (mask.sum(1) + 1e-13)
        loss = loss.mean()

        return loss


class INSPIREDRecModel(BaseModel):
    """

    Attributes:
        item_size: A integer indicating the number of items.

    """

    def __init__(self, opt, device, vocab, side_data):
        """

        Args:
            opt (Config or dict): A dictionary record the hyper parameters.
            device (torch.device): A variable indicating which device to place the data and model.
            vocab (dict): A dictionary record the vocabulary information.
            side_data (dict): A dictionary record the side data.

        """
        self.item_size = vocab['n_entity']

        language = self.opt['language_type']
        resource = resources['bert'][language]
        dpath = os.path.join(opt.pretrain_path, "bert", language)
        super(INSPIREDRecModel, self).__init__(opt, device, dpath, resource)

    def build_model(self):
        # build BERT layer, give the architecture, load pretrained parameters
        self.bert = BertModel.from_pretrained(self.dpath)
        # print(self.item_size)
        self.bert_hidden_size = self.bert.config.hidden_size
        self.mlp = nn.Linear(self.bert_hidden_size, self.item_size)

        # this loss may conduct to some weakness
        self.rec_loss = nn.CrossEntropyLoss()

        logger.debug('[Finish build rec layer]')

    def recommend(self, batch, mode='train'):
        context, mask, y = batch

        bert_embed = self.bert(context, attention_mask=mask).pooler_output

        rec_scores = self.mlp(bert_embed)  # bs, item_size

        rec_loss = self.rec_loss(rec_scores, y)

        return rec_loss, rec_scores


class INSPIREDConvModel(BaseModel):
    """

    Attributes:
        context_truncate: A integer indicating the length of dialogue context.
        response_truncate: A integer indicating the length of dialogue response.
        pad_id: A integer indicating the id of padding token.

    """

    def __init__(self, opt, device, vocab, side_data):
        """

        Args:
            opt (Config or dict): A dictionary record the hyper parameters.
            device (torch.device): A variable indicating which device to place the data and model.
            vocab (dict): A dictionary record the vocabulary information.
            side_data (dict): A dictionary record the side data.

        """
        self.context_truncate = opt['context_truncate']
        self.response_truncate = opt['response_truncate']
        self.pad_id = vocab['pad']
        self.label_smoothing = opt['conv']['label_smoothing'] if 'label_smoothing' in opt['conv'] else -1

        language = self.opt['language_type']
        resource = resources['gpt2'][language]
        dpath = os.path.join(opt.pretrain_path, "gpt2", language)
        super(INSPIREDConvModel, self).__init__(opt, device, dpath, resource)

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
