# @Time   : 2020/12/9
# @Author : Yuanhang Zhou
# @Email  : sdzyh002@gmail.com

# UPDATE:
# @Time   : 2021/1/7, 2020/12/15, 2021/1/4
# @Author : Xiaolei Wang, Yuanhang Zhou, Yuanhang Zhou
# @Email  : wxl1999@foxmail.com, sdzyh002@gmail, sdzyh002@gmail.com

r"""
TGReDial_Conv
=============
References:
    Zhou, Kun, et al. `"Towards Topic-Guided Conversational Recommender System."`_ in COLING 2020.

.. _`"Towards Topic-Guided Conversational Recommender System."`:
   https://www.aclweb.org/anthology/2020.coling-main.365/

"""

import os

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import GPT2LMHeadModel
from transformers import BertModel
from loguru import logger

from crslab.dataset import dataset_language_map
from crslab.model.base import BaseModel
from crslab.model.pretrained_models import resources
from crslab.model.utils.modules.sasrec import SASRec
from crslab.utils import ModelType


class TGConvModel(BaseModel):
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

        language = dataset_language_map[opt['dataset']]
        resource = resources['gpt2'][language]
        dpath = os.path.join(opt.pretrain_path, 'gpt2', language)
        super(TGConvModel, self).__init__(opt, device, dpath, resource)

    def build_model(self):
        """build model"""
        self.model = GPT2LMHeadModel.from_pretrained(self.dpath)
        self.loss = CrossEntropyLoss(ignore_index=self.pad_id)

    def forward(self, batch, mode):
        if mode == 'test' or mode == 'infer':
            enhanced_context = batch[1]
            return self.generate(enhanced_context)
        else:
            enhanced_input_ids = batch[0]
            # torch.tensor's shape = (bs, seq_len, v_s); tuple's length = 12
            lm_logits = self.model(enhanced_input_ids).logits

            # index from 1 to self.reponse_truncate is valid response
            loss = self.calculate_loss(
                lm_logits[:, -self.response_truncate:-1, :],
                enhanced_input_ids[:, -self.response_truncate + 1:])

            pred = torch.max(lm_logits, dim=2)[1]  # [bs, seq_len]
            pred = pred[:, -self.response_truncate:]

            return loss, pred

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

    def generate_bs(self, context, beam=4):
        context = context[..., -self.response_truncate + 1:]
        context_former = context
        batch_size = context.shape[0]
        sequences = [[[list(), 1.0]]] * batch_size
        for i in range(self.response_truncate - 1):
            if sequences != [[[list(), 1.0]]] * batch_size:
                context = []
                for i in range(batch_size):
                    for cand in sequences[i]:
                        text = torch.cat(
                            (context_former[i], torch.tensor(cand[0]).to(self.device)))  # 由于取消了state，与之前的context拼接
                        context.append(text)
                context = torch.stack(context)
            with torch.no_grad():
                outputs = self.model(context)
            last_hidden_state, state = outputs.logits, outputs.past_key_values
            next_token_logits = last_hidden_state[:, -1, :]
            next_token_probs = torch.nn.functional.softmax(next_token_logits)
            topk = torch.topk(next_token_probs, beam, dim=-1)
            probs = topk.values.reshape([batch_size, -1, beam])  # (bs, candidate, beam)
            preds = topk.indices.reshape([batch_size, -1, beam])  # (bs, candidate, beam)

            for j in range(batch_size):
                all_candidates = []
                for n in range(len(sequences[j])):
                    for k in range(beam):
                        seq = sequences[j][n][0]
                        prob = sequences[j][n][1]
                        seq_tmp = seq.copy()
                        seq_tmp.append(preds[j][n][k])
                        candidate = [seq_tmp, prob * probs[j][n][k]]
                        all_candidates.append(candidate)
                ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)
                sequences[j] = ordered[:beam]

        res = []
        for i in range(batch_size):
            res.append(torch.stack(sequences[i][0][0]))
        res = torch.stack(res)
        return res

    def calculate_loss(self, logit, labels):
        """
        Args:
            preds: torch.FloatTensor, shape=(bs, response_truncate, vocab_size)
            labels: torch.LongTensor, shape=(bs, response_truncate)

        """

        loss = self.loss(logit.reshape(-1, logit.size(-1)), labels.reshape(-1))
        return loss


class TGPolicyModel(BaseModel):
    def __init__(self, opt, device, vocab, side_data):
        """

        Args:
            opt (dict): A dictionary record the hyper parameters.
            device (torch.device): A variable indicating which device to place the data and model.
            vocab (dict): A dictionary record the vocabulary information.
            side_data (dict): A dictionary record the side data.

        """
        self.topic_class_num = vocab['n_topic']
        self.n_sent = opt.get('n_sent', 10)

        language = dataset_language_map[opt['dataset']]
        resource = resources['bert'][language]
        dpath = os.path.join(PRETRAIN_PATH, "bert", language)
        super(TGPolicyModel, self).__init__(opt, device, dpath, resource)

    def build_model(self, *args, **kwargs):
        """build model"""
        self.context_bert = BertModel.from_pretrained(self.dpath)
        self.topic_bert = BertModel.from_pretrained(self.dpath)
        self.profile_bert = BertModel.from_pretrained(self.dpath)

        self.bert_hidden_size = self.context_bert.config.hidden_size
        self.state2topic_id = nn.Linear(self.bert_hidden_size * 3,
                                        self.topic_class_num)

        self.loss = nn.CrossEntropyLoss()

    def forward(self, batch, mode):
        # conv_id, message_id, context, context_mask, topic_path_kw, tp_mask, user_profile, profile_mask, y = batch
        context, context_mask, topic_path_kw, tp_mask, user_profile, profile_mask, y = batch

        context_rep = self.context_bert(
            context,
            context_mask).pooler_output  # (bs, hidden_size)

        topic_rep = self.topic_bert(
            topic_path_kw,
            tp_mask).pooler_output  # (bs, hidden_size)

        bs = user_profile.shape[0] // self.n_sent
        profile_rep = self.profile_bert(user_profile, profile_mask).pooler_output  # (bs, word_num, hidden)
        profile_rep = profile_rep.view(bs, self.n_sent, -1)
        profile_rep = torch.mean(profile_rep, dim=1)  # (bs, hidden)

        state_rep = torch.cat((context_rep, topic_rep, profile_rep), dim=1)  # [bs, hidden_size*3]
        topic_scores = self.state2topic_id(state_rep)
        topic_loss = self.loss(topic_scores, y)

        return topic_loss, topic_scores


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
