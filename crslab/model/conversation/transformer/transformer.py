# @Time   : 2020/12/17
# @Author : Yuanhang Zhou
# @Email  : sdzyh002@gmail.com

# UPDATE
# @Time   : 2020/12/29, 2021/1/4
# @Author : Xiaolei Wang, Yuanhang Zhou
# @email  : wxl1999@foxmail.com, sdzyh002@gmail.com

r"""
Transformer
===========
References:
    Zhou, Kun, et al. `"Towards Topic-Guided Conversational Recommender System."`_ in COLING 2020.

.. _`"Towards Topic-Guided Conversational Recommender System."`:
   https://www.aclweb.org/anthology/2020.coling-main.365/

"""

import torch
import torch.nn.functional as F
from loguru import logger
from torch import nn

from crslab.model.base import BaseModel
from crslab.model.utils.functions import edge_to_pyg_format
from crslab.model.utils.modules.transformer import TransformerEncoder, TransformerDecoder


class TransformerModel(BaseModel):
    """

    Attributes:
        vocab_size: A integer indicating the vocabulary size.
        pad_token_idx: A integer indicating the id of padding token.
        start_token_idx: A integer indicating the id of start token.
        end_token_idx: A integer indicating the id of end token.
        token_emb_dim: A integer indicating the dimension of token embedding layer.
        pretrain_embedding: A string indicating the path of pretrained embedding.
        n_word: A integer indicating the number of words.
        n_entity: A integer indicating the number of entities.
        pad_word_idx: A integer indicating the id of word padding.
        pad_entity_idx: A integer indicating the id of entity padding.
        num_bases: A integer indicating the number of bases.
        kg_emb_dim: A integer indicating the dimension of kg embedding.
        n_heads: A integer indicating the number of heads.
        n_layers: A integer indicating the number of layer.
        ffn_size: A integer indicating the size of ffn hidden.
        dropout: A float indicating the drouput rate.
        attention_dropout: A integer indicating the drouput rate of attention layer.
        relu_dropout: A integer indicating the drouput rate of relu layer.
        learn_positional_embeddings: A boolean indicating if we learn the positional embedding.
        embeddings_scale: A boolean indicating if we use the embeddings scale.
        reduction: A boolean indicating if we use the reduction.
        n_positions: A integer indicating the number of position.
        longest_label: A integer indicating the longest length for response generation.

    """

    def __init__(self, opt, device, vocab, side_data):
        """

        Args:
            opt (dict): A dictionary record the hyper parameters.
            device (torch.device): A variable indicating which device to place the data and model.
            vocab (dict): A dictionary record the vocabulary information.
            side_data (dict): A dictionary record the side data.

        """
        # vocab
        self.vocab_size = vocab['vocab_size']
        self.pad_token_idx = vocab['pad']
        self.start_token_idx = vocab['start']
        self.end_token_idx = vocab['end']
        self.token_emb_dim = opt['token_emb_dim']
        self.pretrain_embedding = side_data.get('embedding', None)
        # kg
        self.n_word = vocab['n_word']
        self.n_entity = vocab['n_entity']
        self.pad_word_idx = vocab['pad_word']
        self.pad_entity_idx = vocab['pad_entity']
        entity_kg = side_data['entity_kg']
        self.n_relation = entity_kg['n_relation']
        entity_edges = entity_kg['edge']
        self.entity_edge_idx, self.entity_edge_type = edge_to_pyg_format(entity_edges, 'RGCN')
        self.entity_edge_idx = self.entity_edge_idx.to(device)
        self.entity_edge_type = self.entity_edge_type.to(device)
        word_edges = side_data['word_kg']['edge']
        self.word_edges = edge_to_pyg_format(word_edges, 'GCN').to(device)
        self.num_bases = opt['num_bases']
        self.kg_emb_dim = opt['kg_emb_dim']
        # transformer
        self.n_heads = opt['n_heads']
        self.n_layers = opt['n_layers']
        self.ffn_size = opt['ffn_size']
        self.dropout = opt['dropout']
        self.attention_dropout = opt['attention_dropout']
        self.relu_dropout = opt['relu_dropout']
        self.learn_positional_embeddings = opt['learn_positional_embeddings']
        self.embeddings_scale = opt['embeddings_scale']
        self.reduction = opt['reduction']
        self.n_positions = opt['n_positions']
        self.longest_label = opt.get('longest_label', 1)
        super(TransformerModel, self).__init__(opt, device)

    def build_model(self):
        self._init_embeddings()
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

        logger.debug('[Finish init embeddings]')

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

        self.conv_decoder = TransformerDecoder(
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

        self.conv_loss = nn.CrossEntropyLoss(ignore_index=self.pad_token_idx)

        logger.debug('[Finish build conv layer]')

    def _starts(self, batch_size):
        """Return bsz start tokens."""
        return self.START.detach().expand(batch_size, 1)

    def _decode_forced_with_kg(self, token_encoding, response):
        batch_size, seq_len = response.shape
        start = self._starts(batch_size)
        inputs = torch.cat((start, response[:, :-1]), dim=-1).long()

        dialog_latent, _ = self.conv_decoder(inputs, token_encoding)  # (bs, seq_len, dim)

        gen_logits = F.linear(dialog_latent, self.token_embedding.weight)  # (bs, seq_len, vocab_size)
        preds = gen_logits.argmax(dim=-1)
        return gen_logits, preds

    def _decode_greedy_with_kg(self, token_encoding):
        batch_size = token_encoding[0].shape[0]
        inputs = self._starts(batch_size).long()
        incr_state = None
        logits = []
        for _ in range(self.longest_label):
            dialog_latent, incr_state = self.conv_decoder(inputs, token_encoding, incr_state)
            dialog_latent = dialog_latent[:, -1:, :]  # (bs, 1, dim)

            gen_logits = F.linear(dialog_latent, self.token_embedding.weight)
            preds = gen_logits.argmax(dim=-1).long()
            logits.append(gen_logits)
            inputs = torch.cat((inputs, preds), dim=1)

            finished = ((inputs == self.end_token_idx).sum(dim=-1) > 0).sum().item() == batch_size
            if finished:
                break
        logits = torch.cat(logits, dim=1)
        return logits, inputs

    def _decode_beam_search_with_kg(self, token_encoding, beam=4):
        batch_size = token_encoding[0].shape[0]
        xs = self._starts(batch_size).long().reshape(1, batch_size, -1)
        incr_state = None
        sequences = [[[list(), list(), 1.0]]] * batch_size
        for i in range(self.longest_label):
            # at beginning there is 1 candidate, when i!=0 there are 4 candidates
            if i == 1:
                token_encoding = (token_encoding[0].repeat(beam, 1, 1),
                                  token_encoding[1].repeat(beam, 1, 1))
            if i != 0:
                xs = []
                for d in range(len(sequences[0])):
                    for j in range(batch_size):
                        text = sequences[j][d][0]
                        xs.append(text)
                xs = torch.stack(xs).reshape(beam, batch_size, -1)  # (beam, batch_size, _)

            dialog_latent, incr_state = self.conv_decoder(xs.reshape(len(sequences[0]) * batch_size, -1),
                                                          token_encoding,
                                                          incr_state)
            dialog_latent = dialog_latent[:, -1:, :]  # (bs, 1, dim)
            gen_logits = F.linear(dialog_latent, self.token_embedding.weight)

            logits = gen_logits.reshape(len(sequences[0]), batch_size, 1, -1)
            # turn into probabilities,in case of negative numbers
            probs, preds = torch.nn.functional.softmax(logits).topk(beam, dim=-1)

            # (candeidate, bs, 1 , beam) during first loop, candidate=1, otherwise candidate=beam

            for j in range(batch_size):
                all_candidates = []
                for n in range(len(sequences[j])):
                    for k in range(beam):
                        prob = sequences[j][n][2]
                        logit = sequences[j][n][1]
                        if logit == []:
                            logit_tmp = logits[n][j][0].unsqueeze(0)
                        else:
                            logit_tmp = torch.cat((logit, logits[n][j][0].unsqueeze(0)), dim=0)
                        seq_tmp = torch.cat((xs[n][j].reshape(-1), preds[n][j][0][k].reshape(-1)))
                        candidate = [seq_tmp, logit_tmp, prob * probs[n][j][0][k]]
                        all_candidates.append(candidate)
                ordered = sorted(all_candidates, key=lambda tup: tup[2], reverse=True)
                sequences[j] = ordered[:beam]

            # check if everyone has generated an end token
            all_finished = ((xs == self.end_token_idx).sum(dim=1) > 0).sum().item() == batch_size
            if all_finished:
                break
        logits = torch.stack([seq[0][1] for seq in sequences])
        xs = torch.stack([seq[0][0] for seq in sequences])
        return logits, xs

    def forward(self, batch, mode):
        context_tokens, context_entities, context_words, response = batch

        # encoder-decoder
        tokens_encoding = self.conv_encoder(context_tokens)
        if mode != 'test':
            self.longest_label = max(self.longest_label, response.shape[1])
            logits, preds = self._decode_forced_with_kg(tokens_encoding,
                                                        response)

            logits = logits.view(-1, logits.shape[-1])
            response = response.view(-1)
            loss = self.conv_loss(logits, response)
            return loss, preds
        else:
            logits, preds = self._decode_greedy_with_kg(tokens_encoding)
            return preds
