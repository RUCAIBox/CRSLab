# -*- encoding: utf-8 -*-
# @Time    :   2020/12/4
# @Author  :   Xiaolei Wang
# @email   :   wxl1999@foxmail.com

# UPDATE
# @Time    :   2020/12/4
# @Author  :   Xiaolei Wang
# @email   :   wxl1999@foxmail.com

import torch
import torch.nn.functional as F
from loguru import logger
from torch import nn
from torch_geometric.nn import RGCNConv

from crslab.model.base_model import BaseModel
from crslab.model.layers import SelfAttentionBatch
from crslab.model.transformer import TransformerDecoder, TransformerEncoder
from crslab.model.utils import edge_to_pyg_format


class KBRDModel(BaseModel):
    def __init__(self, opt, device, side_data):
        super(KBRDModel, self).__init__(opt, device)

        self.pad_token_idx = opt['pad_token_idx']
        self.start_token_idx = opt['start_token_idx']
        self.end_token_idx = opt['end_token_idx']
        self.vocab_size = opt['vocab_size']
        self.token_emb_dim = opt.get('token_emb_dim', 300)
        self.pretrain_embedding = side_data.get('embedding', None)

        self.n_entity = opt['n_entity']
        self.n_relation = opt['n_relation']
        self.kg_emb_dim = opt.get('kg_emb_dim', 300)
        self.num_bases = opt.get('num_bases', 8)
        self.edge_idx, self.edge_type = edge_to_pyg_format(side_data['entity_kg'], 'RGCN')
        self.edge_idx = self.edge_idx.to(self.device)
        self.edge_type = self.edge_type.to(self.device)
        self.user_emb_dim = self.kg_emb_dim

        self.n_heads = opt.get('n_heads', 2)
        self.n_layers = opt.get('n_layers', 2)
        self.ffn_size = opt.get('ffn_size', 300)
        self.dropout = opt.get('dropout', 0.1)
        self.attention_dropout = opt.get('attention_dropout', 0.0)
        self.relu_dropout = opt.get('relu_dropout', 0.1)
        self.embeddings_scale = opt.get('embedding_scale', True)
        self.learn_positional_embeddings = opt.get('learn_positional_embeddings', False)
        self.reduction = opt.get('reduction', False)
        self.n_positions = opt.get('n_positions', 1024)
        self.longest_label = opt.get('longest_label', 1)

        self.build_model()

    def build_model(self, *args, **kwargs):
        self._build_embedding()
        self._build_kg_layer()
        self._build_recommendation_layer()
        self._build_conversation_layer()

    def _build_embedding(self):
        if self.pretrain_embedding is not None:
            self.token_embedding = nn.Embedding.from_pretrained(
                torch.as_tensor(self.pretrain_embedding, dtype=torch.float), freeze=False,
                padding_idx=self.pad_token_idx)
        else:
            self.token_embedding = nn.Embedding(self.vocab_size, self.token_emb_dim, self.pad_token_idx)
            nn.init.normal_(self.token_embedding.weight, mean=0, std=self.kg_emb_dim ** -0.5)
            nn.init.constant_(self.token_embedding.weight[self.pad_token_idx], 0)
        logger.debug('[Build embedding]')

    def _build_kg_layer(self):
        self.kg_encoder = RGCNConv(self.n_entity, self.kg_emb_dim, self.n_relation, num_bases=self.num_bases)
        self.kg_attn = SelfAttentionBatch(self.kg_emb_dim, self.kg_emb_dim)
        logger.debug('[Build kg layer]')

    def _build_recommendation_layer(self):
        self.rec_bias = nn.Linear(self.kg_emb_dim, self.n_entity)
        self.rec_loss = nn.CrossEntropyLoss()
        logger.debug('[Build recommendation layer]')

    def _build_conversation_layer(self):
        self.register_buffer('START', torch.tensor([self.start_token_idx], dtype=torch.long))
        self.dialog_encoder = TransformerEncoder(
            self.n_heads,
            self.n_layers,
            self.token_emb_dim,
            self.ffn_size,
            self.vocab_size,
            self.token_embedding,
            self.dropout,
            self.attention_dropout,
            self.relu_dropout,
            self.pad_token_idx,
            self.learn_positional_embeddings,
            self.embeddings_scale,
            self.reduction,
            self.n_positions
        )
        self.decoder = TransformerDecoder(
            self.n_heads,
            self.n_layers,
            self.token_emb_dim,
            self.ffn_size,
            self.vocab_size,
            self.token_embedding,
            self.dropout,
            self.attention_dropout,
            self.relu_dropout,
            self.embeddings_scale,
            self.learn_positional_embeddings,
            self.pad_token_idx,
            self.n_positions
        )
        self.user_proj_1 = nn.Linear(self.user_emb_dim, 512)
        self.user_proj_2 = nn.Linear(512, self.vocab_size)
        self.conv_loss = nn.CrossEntropyLoss(ignore_index=self.pad_token_idx, reduction='sum')
        logger.debug('[Build conversation layer]')

    def encode_user(self, entity_lists, kg_embedding):
        user_repr_list = []
        for entity_list in entity_lists:
            if not entity_list:
                user_repr_list.append(torch.zeros(self.user_emb_dim, device=self.device))
                continue
            user_repr = kg_embedding[entity_list]
            user_repr = self.kg_attn(user_repr)
            user_repr_list.append(user_repr)
        return torch.stack(user_repr_list, dim=0)  # (bs, dim)

    def recommend(self, batch):
        context_entities, item = batch['context_entities'], batch['item']
        kg_embedding = self.kg_encoder(None, self.edge_idx, self.edge_type)
        user_embedding = self.encode_user(context_entities, kg_embedding)
        scores = F.linear(user_embedding, kg_embedding, self.rec_bias.bias)
        loss = self.rec_loss(scores, item)
        return loss, scores

    def _starts(self, batch_size):
        """Return bsz start tokens."""
        return self.START.detach().expand(batch_size, 1)

    def decode_forced(self, encoder_states, user_embedding, resp):
        bsz = resp.size(0)
        seqlen = resp.size(1)
        inputs = resp.narrow(1, 0, seqlen - 1)
        inputs = torch.cat([self._starts(bsz), inputs], 1)
        latent, _ = self.decoder(inputs, encoder_states)
        token_logits = F.linear(latent, self.token_embedding.weight)
        user_logits = self.user_proj_2(torch.relu(self.user_proj_1(user_embedding))).unsqueeze(1)
        sum_logits = token_logits + user_logits
        _, preds = sum_logits.max(dim=-1)
        return sum_logits, preds

    def decode_greedy(self, encoder_states, user_embedding):
        bsz = encoder_states[0].shape[0]
        xs = self._starts(bsz)
        incr_state = None
        logits = []
        for i in range(self.longest_label):
            scores, incr_state = self.decoder(xs, encoder_states, incr_state)
            scores = scores[:, -1:, :]
            token_logits = F.linear(scores, self.token_embedding.weight)
            user_logits = self.user_proj_2(torch.relu(self.user_proj_1(user_embedding))).unsqueeze(1)
            sum_logits = token_logits + user_logits
            _, preds = sum_logits.max(dim=-1)
            logits.append(scores)
            xs = torch.cat([xs, preds], dim=1)
            # check if everyone has generated an end token
            all_finished = ((xs == self.end_token_idx).sum(dim=1) > 0).sum().item() == bsz
            if all_finished:
                break
        logits = torch.cat(logits, 1)
        return logits, xs

    def converse(self, batch, mode='train'):
        context_tokens, context_entities, response = batch['context_tokens'], batch['context_entities'], batch[
            'response']
        kg_embedding = self.kg_encoder(None, self.edge_idx, self.edge_type)
        user_embedding = self.encode_user(context_entities, kg_embedding)
        encoder_state = self.dialog_encoder(context_tokens)
        if mode != 'test':
            self.longest_label = max(self.longest_label, response.shape[1])
            logits, preds = self.decode_forced(encoder_state, user_embedding, response)
            logits = logits.view(-1, logits.shape[-1])
            labels = response.view(-1)
            notnull = labels.ne(self.pad_token_idx)
            target_tokens = notnull.long().sum().item()
            return self.conv_loss(logits, labels) / target_tokens, preds
        else:
            _, preds = self.decode_greedy(encoder_state, user_embedding)
            return preds
