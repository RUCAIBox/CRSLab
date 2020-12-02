# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/24, 2020/12/2
# @Author : Kun Zhou, Xiaolei Wang
# @Email  : francis_kun_zhou@163.com, wxl1999@foxmail.com

import os

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from torch import nn
from torch_geometric.nn import GCNConv, RGCNConv

from crslab.config.config import DATA_PATH
from crslab.data import DownloadableFile, build
from crslab.model.base_model import BaseModel
from crslab.model.layers import SelfAttentionSeq, GateLayer
from crslab.model.transformer import TransformerEncoder, TransformerDecoderKG
from crslab.model.utils import edge_to_pyg_format


class KGSFModel(BaseModel):
    r"""
    S3Rec is the first work to incorporate self-supervised learning in
    sequential recommendation.

    NOTE:
        Under this framework, we need reconstruct the pretraining data,
        which would affect the pre-training speed.
    """

    def __init__(self, opt, device, side_data):
        super(KGSFModel, self).__init__(opt, device)
        # vocab
        self.vocab_size = self.opt['vocab_size']
        self.pad_token_idx = self.opt['pad_token_idx']
        self.start_token_idx = self.opt['start_token_idx']
        self.end_token_idx = self.opt['end_token_idx']
        self.token_emb_dim = self.opt['token_emb_dim']
        self.pretrain_embedding = side_data.get('embedding', None)
        # kg
        self.n_word = self.opt['n_word']
        self.n_entity = self.opt['n_entity']
        self.n_relation = self.opt['n_relation']
        self.kg_emb_dim = self.opt['kg_emb_dim']
        self.pad_word_idx = self.opt['pad_word_idx']
        self.pad_entity_idx = self.opt['pad_entity_idx']
        entity_edges, word_edges = side_data['entity_kg'], side_data['word_kg']
        self.entity_edges = edge_to_pyg_format(entity_edges, device)
        self.word_edges = edge_to_pyg_format(word_edges, device, 'GCN')
        self.num_bases = self.opt['num_bases']
        # transformer
        self.n_heads = self.opt['n_heads']
        self.n_layers = self.opt['n_layers']
        self.ffn_size = self.opt['ffn_size']
        self.dropout = self.opt['dropout']
        self.attention_dropout = self.opt['attention_dropout']
        self.relu_dropout = self.opt['relu_dropout']
        self.learn_positional_embeddings = self.opt['learn_positional_embeddings']
        self.embeddings_scale = self.opt['embeddings_scale']
        self.reduction = self.opt['reduction']
        self.n_positions = self.opt['n_positions']
        self.response_truncate = self.opt.get('response_truncate', 20)
        # copy mask
        dpath = os.path.join(DATA_PATH, "kgsf")
        dfile = DownloadableFile('1zrszs2EcNlim3l7O0BH6XbalLMeUcMFv', 'kgsf.zip',
                                 'f627841644a184079acde1b0185e3a223945061c3a591f4bc0d7f62e7263f548',
                                 from_google=True)
        build(dpath, dfile)
        self.copy_mask = torch.as_tensor(np.load(os.path.join(dpath, "copy_mask.npy")).astype(bool), device=self.device)

        self.build_model()

    def build_model(self):
        self._init_embeddings()
        self._build_kg_layer()
        self._build_infomax_layer()
        self._build_recommendation_layer()
        self._build_conversation_layer()

    def _init_embeddings(self):
        if self.pretrain_embedding is not None:
            self.token_embedding = nn.Embedding.from_pretrained(
                torch.as_tensor(self.pretrain_embedding, dtype=torch.float), freeze=False,
                padding_idx=self.pad_token_idx)
        else:
            self.token_embedding = nn.Embedding(self.vocab_size, self.token_emb_dim, self.pad_token_idx)
            nn.init.normal_(self.token_embedding.weight, mean=0, std=self.kg_emb_dim ** -0.5)
            nn.init.constant_(self.token_embedding.weight[self.pad_token_idx], 0)

        self.word_kg_embedding = nn.Embedding(self.n_word, self.kg_emb_dim, self.pad_word_idx)
        nn.init.normal_(self.word_kg_embedding.weight, mean=0, std=self.kg_emb_dim ** -0.5)
        nn.init.constant_(self.word_kg_embedding.weight[self.pad_word_idx], 0)

        logger.debug('[Finish init embeddings]')

    def _build_kg_layer(self):
        # db encoder
        self.entity_encoder = RGCNConv(self.n_entity, self.kg_emb_dim, self.n_relation, self.num_bases)
        self.entity_self_attn = SelfAttentionSeq(self.kg_emb_dim, self.kg_emb_dim)

        # concept encoder
        self.word_encoder = GCNConv(self.kg_emb_dim, self.kg_emb_dim)
        self.word_self_attn = SelfAttentionSeq(self.kg_emb_dim, self.kg_emb_dim)

        # gate mechanism
        self.gate_layer = GateLayer(self.kg_emb_dim)
        logger.debug('[Finish build kg layer]')

    def _build_infomax_layer(self):
        self.infomax_norm = nn.Linear(self.kg_emb_dim, self.kg_emb_dim)
        self.infomax_bias = nn.Linear(self.kg_emb_dim, self.n_entity)
        self.infomax_loss = nn.MSELoss(reduction='sum')
        logger.debug('[Finish build infomax layer]')

    def _build_recommendation_layer(self):
        self.rec_bias = nn.Linear(self.kg_emb_dim, self.n_entity)
        self.rec_loss = nn.CrossEntropyLoss()
        logger.debug('[Finish build rec layer]')

    def _build_conversation_layer(self):
        self.register_buffer('START', torch.tensor([self.start_token_idx], dtype=torch.long))
        self.conv_encoder = TransformerEncoder(
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            embedding_size=self.token_emb_dim,
            ffn_size=self.ffn_size,
            vocabulary_size=self.vocab_size,
            embedding=self.token_embedding,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            relu_dropout=self.relu_dropout,
            padding_idx=self.pad_token_idx,
            learn_positional_embeddings=self.learn_positional_embeddings,
            embeddings_scale=self.embeddings_scale,
            reduction=self.reduction,
            n_positions=self.n_positions,
        )

        self.conv_entity_norm = nn.Linear(self.kg_emb_dim, self.ffn_size)
        self.conv_entity_attn_norm = nn.Linear(self.kg_emb_dim, self.ffn_size)
        self.conv_word_norm = nn.Linear(self.kg_emb_dim, self.ffn_size)
        self.conv_word_attn_norm = nn.Linear(self.kg_emb_dim, self.ffn_size)

        self.copy_norm = nn.Linear(self.ffn_size * 3, self.token_emb_dim)
        self.copy_output = nn.Linear(self.token_emb_dim, self.vocab_size)

        self.conv_decoder = TransformerDecoderKG(
            self.n_heads, self.n_layers, self.token_emb_dim, self.ffn_size, self.vocab_size,
            embedding=self.token_embedding,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            relu_dropout=self.relu_dropout,
            embeddings_scale=self.embeddings_scale,
            learn_positional_embeddings=self.learn_positional_embeddings,
            padding_idx=self.pad_token_idx,
            n_positions=self.n_positions
        )
        self.conv_loss = nn.CrossEntropyLoss(reduction="sum", ignore_index=self.pad_token_idx)
        logger.debug('[Finish build conv layer]')

    def pretrain_infomax(self, batch):
        """
        words: (batch_size, word_length)
        entity_labels: (batch_size, n_entity)
        """
        words, entity_labels = batch

        loss_mask = torch.sum(entity_labels)
        if loss_mask.item() == 0:
            return None

        entity_graph_representations = self.entity_encoder(None, self.entity_edges[0], self.entity_edges[1])
        word_graph_representations = self.word_encoder(self.word_kg_embedding.weight, self.word_edges)

        word_representations = word_graph_representations[words]
        word_padding_mask = words.eq(self.pad_word_idx)  # (bs, seq_len)

        word_attn_rep = self.word_self_attn(word_representations, word_padding_mask)
        word_info_rep = self.infomax_norm(word_attn_rep)  # (bs, dim)
        info_predict = F.linear(word_info_rep, entity_graph_representations, self.infomax_bias.bias)  # (bs, #entity)
        loss = self.infomax_loss(info_predict, entity_labels) / loss_mask
        return loss

    def recommender(self, batch, mode='train'):
        """
        context_entities: (batch_size, entity_length)
        context_words: (batch_size, word_length)
        movie: (batch_size)
        """
        context_entities, context_words, entities, movie = batch

        entity_graph_representations = self.entity_encoder(None, self.entity_edges[0], self.entity_edges[1])
        word_graph_representations = self.word_encoder(self.word_kg_embedding.weight, self.word_edges)

        entity_padding_mask = context_entities.eq(self.pad_entity_idx)  # (bs, entity_len)
        word_padding_mask = context_words.eq(self.pad_word_idx)  # (bs, word_len)

        entity_representations = entity_graph_representations[context_entities]
        word_representations = word_graph_representations[context_words]

        entity_attn_rep = self.entity_self_attn(entity_representations, entity_padding_mask)
        word_attn_rep = self.word_self_attn(word_representations, word_padding_mask)

        user_rep = self.gate_layer(entity_attn_rep, word_attn_rep)
        rec_scores = F.linear(user_rep, entity_graph_representations, self.rec_bias.bias)  # (bs, #entity)

        rec_loss = self.rec_loss(rec_scores, movie)

        info_loss_mask = torch.sum(entities)
        if info_loss_mask.item() == 0:
            info_loss = None
        else:
            word_info_rep = self.infomax_norm(word_attn_rep)  # (bs, dim)
            info_predict = F.linear(word_info_rep, entity_graph_representations,
                                    self.infomax_bias.bias)  # (bs, #entity)
            info_loss = self.infomax_loss(info_predict, entities) / info_loss_mask

        return rec_loss, info_loss, rec_scores

    def stem_conv_parameters(self):
        for params in [self.word_kg_embedding.parameters(), self.entity_encoder, self.entity_self_attn,
                       self.word_encoder, self.word_self_attn, self.gate_layer,
                       self.infomax_bias, self.infomax_norm, self.rec_bias]:
            for p in params:
                p.requires_grad = False

    def _starts(self, batch_size):
        """Return bsz start tokens."""
        return self.START.detach().expand(batch_size, 1)

    def _KG_transformer_decode_forced(self, token_encoding, entity_reps, entity_emb_attn, entity_mask,
                                      word_reps, word_emb_attn, word_mask, response):
        batch_size, seq_len = response.shape
        start = self._starts(batch_size)
        inputs = torch.cat((start, response[:, :-1]), dim=-1).long()

        dialog_latent, _ = self.conv_decoder(inputs, token_encoding, word_reps, word_mask,
                                             entity_reps, entity_mask)  # (bs, seq_len, dim)
        entity_latent = entity_emb_attn.unsqueeze(1).expand(-1, seq_len, -1)
        word_latent = word_emb_attn.unsqueeze(1).expand(-1, seq_len, -1)
        copy_latent = self.copy_norm(
            torch.cat((entity_latent, word_latent, dialog_latent), dim=-1))  # (bs, seq_len, dim)

        copy_logits = self.copy_output(copy_latent) * self.copy_mask.unsqueeze(0).unsqueeze(
            0)  # (bs, seq_len, vocab_size)
        gen_logits = F.linear(dialog_latent, self.token_embedding.weight)  # (bs, seq_len, vocab_size)
        sum_logits = copy_logits + gen_logits
        preds = sum_logits.argmax(dim=-1)
        return sum_logits, preds

    def _KG_transformer_decode_greedy(self, token_encoding, entity_reps, entity_emb_attn, entity_mask,
                                      word_reps, word_emb_attn, word_mask):
        batch_size = token_encoding[0].shape[0]
        inputs = self._starts(batch_size).long()
        incr_state = None
        logits = []
        for _ in range(self.response_truncate):
            dialog_latent, incr_state = self.conv_decoder(inputs, token_encoding, word_reps, word_mask,
                                                          entity_reps, entity_mask, incr_state)
            dialog_latent = dialog_latent[:, -1:, :]  # (bs, 1, dim)
            db_latent = entity_emb_attn.unsqueeze(1)
            concept_latent = word_emb_attn.unsqueeze(1)
            copy_latent = self.copy_norm(torch.cat((db_latent, concept_latent, dialog_latent), dim=-1))

            copy_logits = self.copy_output(copy_latent) * self.copy_mask.unsqueeze(0).unsqueeze(0)
            gen_logits = F.linear(dialog_latent, self.token_embedding.weight)
            sum_logits = copy_logits + gen_logits
            preds = sum_logits.argmax(dim=-1).long()
            logits.append(sum_logits)
            inputs = torch.cat((inputs, preds), dim=1)

            finished = ((inputs == self.end_token_idx).sum(dim=-1) > 0).sum().item() == batch_size
            if finished:
                break
        logits = torch.cat(logits, dim=1)
        return logits, inputs

    def conversation(self, batch, mode='train'):
        context_tokens, context_entities, context_words, response = batch

        entity_graph_representations = self.entity_encoder(None, self.entity_edges[0], self.entity_edges[1])
        word_graph_representations = self.word_encoder(self.word_kg_embedding.weight, self.word_edges)

        entity_padding_mask = context_entities.eq(self.pad_entity_idx)  # (bs, entity_len)
        word_padding_mask = context_words.eq(self.pad_word_idx)  # (bs, seq_len)

        entity_representations = entity_graph_representations[context_entities]
        word_representations = word_graph_representations[context_words]

        entity_attn_rep = self.entity_self_attn(entity_representations, entity_padding_mask)
        word_attn_rep = self.word_self_attn(word_representations, word_padding_mask)

        # encoder-decoder
        tokens_encoding = self.conv_encoder(context_tokens)
        conv_entity_emb = self.conv_entity_attn_norm(entity_attn_rep)
        conv_word_emb = self.conv_word_attn_norm(word_attn_rep)
        conv_entity_reps = self.conv_entity_norm(entity_representations)
        conv_word_reps = self.conv_word_norm(word_representations)
        if mode != 'test':
            logits, preds = self._KG_transformer_decode_forced(tokens_encoding, conv_entity_reps, conv_entity_emb,
                                                               entity_padding_mask,
                                                               conv_word_reps, conv_word_emb, word_padding_mask,
                                                               response)

            logits = logits.view(-1, logits.shape[-1])
            response = response.view(-1)
            response_mask = response.ne(self.pad_token_idx)

            loss = self.conv_loss(logits, response)
            return loss / torch.sum(response_mask), preds
        else:
            logits, preds = self._KG_transformer_decode_greedy(tokens_encoding, conv_entity_reps, conv_entity_emb,
                                                               entity_padding_mask,
                                                               conv_word_reps, conv_word_emb, word_padding_mask)
            return preds
