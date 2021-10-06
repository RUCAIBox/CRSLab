# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/24, 2020/12/29, 2021/1/4
# @Author : Kun Zhou, Xiaolei Wang, Yuanhang Zhou
# @Email  : francis_kun_zhou@163.com, wxl1999@foxmail.com, sdzyh002@gmail.com

r"""
KGSF
====
References:
    Zhou, Kun, et al. `"Improving Conversational Recommender Systems via Knowledge Graph based Semantic Fusion."`_ in KDD 2020.

.. _`"Improving Conversational Recommender Systems via Knowledge Graph based Semantic Fusion."`:
   https://dl.acm.org/doi/abs/10.1145/3394486.3403143

"""

import os

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from torch import nn
from torch_geometric.nn import GCNConv, RGCNConv

from crslab.config import MODEL_PATH
from crslab.model.base import BaseModel
from crslab.model.utils.functions import edge_to_pyg_format
from crslab.model.utils.modules.attention import SelfAttentionSeq
from crslab.model.utils.modules.transformer import TransformerEncoder
from crslab.download import DownloadableFile
from crslab.model.utils.modules.transformer import MultiHeadAttention, TransformerFFN, _create_selfattn_mask, \
    _normalize, \
    create_position_codes

resources = {
    'ReDial': {
        'version': '0.2',
        'file': DownloadableFile(
            'https://pkueducn-my.sharepoint.com/:u:/g/personal/franciszhou_pku_edu_cn/EXl2bhU82O5Itp9K4Mh41mYB69BKPEvMcKwZRstfYZUB1g?download=1',
            'kgsf_redial.zip',
            'f627841644a184079acde1b0185e3a223945061c3a591f4bc0d7f62e7263f548',
        ),
    },
    'TGReDial': {
        'version': '0.2',
        'file': DownloadableFile(
            'https://pkueducn-my.sharepoint.com/:u:/g/personal/franciszhou_pku_edu_cn/ETzJ0-QnguRKiKO_ktrTDZQBZHKom4-V5SJ9mhesfXzrWQ?download=1',
            'kgsf_tgredial.zip',
            'c9d054b653808795035f77cb783227e6e9a938e5bedca4d7f88c6dfb539be5d1',
        ),
    },
    'GoRecDial': {
        'version': '0.1',
        'file': DownloadableFile(
            'https://pkueducn-my.sharepoint.com/:u:/g/personal/franciszhou_pku_edu_cn/EUfPcGfLHAJPj-F3Mr79CF4Bc5sZXKk-jysutrjiRcQvCg?download=1',
            'kgsf_gorecdial.zip',
            '9794abf12b5d6773d867556685da14d951d42f64a5c4781af7d6fb720e87ec4f',
        )
    },
    'OpenDialKG': {
        'version': '0.1',
        'file': DownloadableFile(
            'https://pkueducn-my.sharepoint.com/:u:/g/personal/franciszhou_pku_edu_cn/EQgebOKypMlPr18KJ6uGeDABtqTbMQYVYNWNR_DaAZ1Wvg?download=1',
            'kgsf_opendialkg.zip',
            '89b785b23478b1d91d6ab4f34a3658e82b52dcbb73828713a9b369fa49db9e61'
        )
    },
    'Inspired': {
        'version': '0.1',
        'file': DownloadableFile(
            'https://pkueducn-my.sharepoint.com/:u:/g/personal/franciszhou_pku_edu_cn/EXQGUxjGQ-ZKpzTnUYOMavABMUAxb0JwkiIMAPp5DIvsNw?download=1',
            'kgsf_inspired.zip',
            '23dfc031a3c71f2a52e29fe0183e1a501771b8d431852102ba6fd83d971f928d'
        )
    },
    'DuRecDial': {
        'version': '0.1',
        'file': DownloadableFile(
            'https://pkueducn-my.sharepoint.com/:u:/g/personal/franciszhou_pku_edu_cn/Ed9-qLkK0bNCk5AAvJpWU3cBC-cXks-6JlclYp08AFovyw?download=1',
            'kgsf_durecdial.zip',
            'f9a39c2382efe88d80ef14d7db8b4cbaf3a6eb92a33e018dfc9afba546ba08ef'
        )
    }
}


class KGSFModel(BaseModel):
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
        dropout: A float indicating the dropout rate.
        attention_dropout: A integer indicating the dropout rate of attention layer.
        relu_dropout: A integer indicating the dropout rate of relu layer.
        learn_positional_embeddings: A boolean indicating if we learn the positional embedding.
        embeddings_scale: A boolean indicating if we use the embeddings scale.
        reduction: A boolean indicating if we use the reduction.
        n_positions: A integer indicating the number of position.
        response_truncate = A integer indicating the longest length for response generation.
        pretrained_embedding: A string indicating the path of pretrained embedding.

    """

    def __init__(self, opt, device, other_data):
        """

        Args:
            opt (dict): A dictionary record the hyper parameters.
            device (torch.device): A variable indicating which device to place the data and model.
            other_data (dict): A dictionary record the other data.

        """
        self.device = device
        self.gpu = opt.get("gpu", [-1])
        # vocab
        self.vocab_size = other_data['vocab']['vocab_size']
        self.pad_token_idx = other_data['vocab']['pad']
        self.start_token_idx = other_data['vocab']['start']
        self.end_token_idx = other_data['vocab']['end']
        self.token_emb_dim = opt['token_emb_dim']
        self.pretrained_embedding = other_data.get('embedding', None)
        # kg
        self.n_word = other_data['vocab']['n_word']
        self.n_entity = other_data['vocab']['n_entity']
        self.pad_word_idx = other_data['vocab']['pad_word']
        self.pad_entity_idx = other_data['vocab']['pad_entity']
        entity_kg = other_data['entity_kg']
        self.n_relation = entity_kg['n_relation']
        entity_edges = entity_kg['edge']
        self.entity_edge_idx, self.entity_edge_type = edge_to_pyg_format(entity_edges, 'RGCN')
        self.entity_edge_idx = self.entity_edge_idx.to(device)
        self.entity_edge_type = self.entity_edge_type.to(device)
        word_edges = other_data['word_kg']['edge']

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
        self.response_truncate = opt.get('response_truncate', 20)
        # copy mask
        dataset = opt['dataset']
        dpath = os.path.join(MODEL_PATH, "kgsf", dataset)
        resource = resources[dataset]
        super(KGSFModel, self).__init__(opt, device, dpath, resource)

    def build_model(self):
        self._init_embeddings()
        self._build_kg_layer()
        self._build_infomax_layer()
        self._build_recommendation_layer()
        self._build_conversation_layer()

    def _init_embeddings(self):
        if self.pretrained_embedding is not None:
            self.token_embedding = nn.Embedding.from_pretrained(
                torch.as_tensor(self.pretrained_embedding, dtype=torch.float), freeze=False,
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
        self.copy_mask = torch.as_tensor(np.load(os.path.join(self.dpath, "copy_mask.npy")).astype(bool),
                                         ).to(self.device)

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
        self.conv_loss = nn.CrossEntropyLoss(ignore_index=self.pad_token_idx)

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

        entity_graph_representations = self.entity_encoder(None, self.entity_edge_idx, self.entity_edge_type)
        word_graph_representations = self.word_encoder(self.word_kg_embedding.weight, self.word_edges)

        word_representations = word_graph_representations[words]
        word_padding_mask = words.eq(self.pad_word_idx)  # (bs, seq_len)

        word_attn_rep = self.word_self_attn(word_representations, word_padding_mask)
        word_info_rep = self.infomax_norm(word_attn_rep)  # (bs, dim)
        info_predict = F.linear(word_info_rep, entity_graph_representations, self.infomax_bias.bias)  # (bs, #entity)
        loss = self.infomax_loss(info_predict, entity_labels) / loss_mask
        return loss

    def recommend(self, batch, mode):
        """
        context_entities: (batch_size, entity_length)
        context_words: (batch_size, word_length)
        movie: (batch_size)
        """
        context_entities, context_words, entities, movie = batch

        entity_graph_representations = self.entity_encoder(None, self.entity_edge_idx, self.entity_edge_type)
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

    def freeze_parameters(self):
        freeze_models = [self.word_kg_embedding, self.entity_encoder, self.entity_self_attn, self.word_encoder,
                         self.word_self_attn, self.gate_layer, self.infomax_bias, self.infomax_norm, self.rec_bias]
        for model in freeze_models:
            for p in model.parameters():
                p.requires_grad = False

    def _starts(self, batch_size):
        """Return bsz start tokens."""
        return self.START.detach().expand(batch_size, 1)

    def _decode_forced_with_kg(self, token_encoding, entity_reps, entity_emb_attn, entity_mask,
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

    def _decode_greedy_with_kg(self, token_encoding, entity_reps, entity_emb_attn, entity_mask,
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

    def _decode_beam_search_with_kg(self, token_encoding, entity_reps, entity_emb_attn, entity_mask,
                                    word_reps, word_emb_attn, word_mask, beam=4):
        batch_size = token_encoding[0].shape[0]
        inputs = self._starts(batch_size).long().reshape(1, batch_size, -1)
        incr_state = None

        sequences = [[[list(), list(), 1.0]]] * batch_size
        for i in range(self.response_truncate):
            if i == 1:
                token_encoding = (token_encoding[0].repeat(beam, 1, 1),
                                  token_encoding[1].repeat(beam, 1, 1))
                entity_reps = entity_reps.repeat(beam, 1, 1)
                entity_emb_attn = entity_emb_attn.repeat(beam, 1)
                entity_mask = entity_mask.repeat(beam, 1)
                word_reps = word_reps.repeat(beam, 1, 1)
                word_emb_attn = word_emb_attn.repeat(beam, 1)
                word_mask = word_mask.repeat(beam, 1)

            # at beginning there is 1 candidate, when i!=0 there are 4 candidates
            if i != 0:
                inputs = []
                for d in range(len(sequences[0])):
                    for j in range(batch_size):
                        text = sequences[j][d][0]
                        inputs.append(text)
                inputs = torch.stack(inputs).reshape(beam, batch_size, -1)  # (beam, batch_size, _)

            with torch.no_grad():
                dialog_latent, incr_state = self.conv_decoder(
                    inputs.reshape(len(sequences[0]) * batch_size, -1),
                    token_encoding, word_reps, word_mask,
                    entity_reps, entity_mask, incr_state
                )
                dialog_latent = dialog_latent[:, -1:, :]  # (bs, 1, dim)
                db_latent = entity_emb_attn.unsqueeze(1)
                concept_latent = word_emb_attn.unsqueeze(1)
                copy_latent = self.copy_norm(torch.cat((db_latent, concept_latent, dialog_latent), dim=-1))

                copy_logits = self.copy_output(copy_latent) * self.copy_mask.unsqueeze(0).unsqueeze(0)
                gen_logits = F.linear(dialog_latent, self.token_embedding.weight)
                sum_logits = copy_logits + gen_logits

            logits = sum_logits.reshape(len(sequences[0]), batch_size, 1, -1)
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
                        seq_tmp = torch.cat((inputs[n][j].reshape(-1), preds[n][j][0][k].reshape(-1)))
                        candidate = [seq_tmp, logit_tmp, prob * probs[n][j][0][k]]
                        all_candidates.append(candidate)
                ordered = sorted(all_candidates, key=lambda tup: tup[2], reverse=True)
                sequences[j] = ordered[:beam]

            # check if everyone has generated an end token
            all_finished = ((inputs == self.end_token_idx).sum(dim=1) > 0).sum().item() == batch_size
            if all_finished:
                break
        logits = torch.stack([seq[0][1] for seq in sequences])
        inputs = torch.stack([seq[0][0] for seq in sequences])
        return logits, inputs

    def converse(self, batch, mode):
        context_tokens, context_entities, context_words, response = batch

        entity_graph_representations = self.entity_encoder(None, self.entity_edge_idx, self.entity_edge_type)
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
            logits, preds = self._decode_forced_with_kg(tokens_encoding, conv_entity_reps, conv_entity_emb,
                                                        entity_padding_mask,
                                                        conv_word_reps, conv_word_emb, word_padding_mask,
                                                        response)

            logits = logits.view(-1, logits.shape[-1])
            response = response.view(-1)
            loss = self.conv_loss(logits, response)
            return loss, preds
        else:
            logits, preds = self._decode_greedy_with_kg(tokens_encoding, conv_entity_reps, conv_entity_emb,
                                                        entity_padding_mask,
                                                        conv_word_reps, conv_word_emb, word_padding_mask)
            return preds

    def forward(self, batch, stage, mode):
        if len(self.gpu) >= 2:
            #  forward function operates on different gpus, the weight of graph network need to be copied to other gpu
            self.entity_edge_idx = self.entity_edge_idx.cuda(torch.cuda.current_device())
            self.entity_edge_type = self.entity_edge_type.cuda(torch.cuda.current_device())
            self.word_edges = self.word_edges.cuda(torch.cuda.current_device())
            self.copy_mask = torch.as_tensor(np.load(os.path.join(self.dpath, "copy_mask.npy")).astype(bool),
                                             ).cuda(torch.cuda.current_device())
        if stage == "pretrain":
            loss = self.pretrain_infomax(batch)
        elif stage == "rec":
            loss = self.recommend(batch, mode)
        elif stage == "conv":
            loss = self.converse(batch, mode)
        return loss


class GateLayer(nn.Module):
    def __init__(self, input_dim):
        super(GateLayer, self).__init__()
        self._norm_layer1 = nn.Linear(input_dim * 2, input_dim)
        self._norm_layer2 = nn.Linear(input_dim, 1)

    def forward(self, input1, input2):
        norm_input = self._norm_layer1(torch.cat([input1, input2], dim=-1))
        gate = torch.sigmoid(self._norm_layer2(norm_input))  # (bs, 1)
        gated_emb = gate * input1 + (1 - gate) * input2  # (bs, dim)
        return gated_emb


class TransformerDecoderLayerKG(nn.Module):
    def __init__(
            self,
            n_heads,
            embedding_size,
            ffn_size,
            attention_dropout=0.0,
            relu_dropout=0.0,
            dropout=0.0,
    ):
        super().__init__()
        self.dim = embedding_size
        self.ffn_dim = ffn_size
        self.dropout = nn.Dropout(p=dropout)

        self.self_attention = MultiHeadAttention(
            n_heads, embedding_size, dropout=attention_dropout
        )
        self.norm1 = nn.LayerNorm(embedding_size)

        self.encoder_attention = MultiHeadAttention(
            n_heads, embedding_size, dropout=attention_dropout
        )
        self.norm2 = nn.LayerNorm(embedding_size)

        self.encoder_db_attention = MultiHeadAttention(
            n_heads, embedding_size, dropout=attention_dropout
        )
        self.norm2_db = nn.LayerNorm(embedding_size)

        self.encoder_kg_attention = MultiHeadAttention(
            n_heads, embedding_size, dropout=attention_dropout
        )
        self.norm2_kg = nn.LayerNorm(embedding_size)

        self.ffn = TransformerFFN(embedding_size, ffn_size, relu_dropout=relu_dropout)
        self.norm3 = nn.LayerNorm(embedding_size)

    def forward(self, x, encoder_output, encoder_mask, kg_encoder_output, kg_encoder_mask, db_encoder_output,
                db_encoder_mask):
        decoder_mask = _create_selfattn_mask(x)
        # first self attn
        residual = x
        # don't peak into the future!
        x = self.self_attention(query=x, mask=decoder_mask)
        x = self.dropout(x)  # --dropout
        x = x + residual
        x = _normalize(x, self.norm1)

        residual = x
        x = self.encoder_db_attention(
            query=x,
            key=db_encoder_output,
            value=db_encoder_output,
            mask=db_encoder_mask
        )
        x = self.dropout(x)  # --dropout
        x = residual + x
        x = _normalize(x, self.norm2_db)

        residual = x
        x = self.encoder_kg_attention(
            query=x,
            key=kg_encoder_output,
            value=kg_encoder_output,
            mask=kg_encoder_mask
        )
        x = self.dropout(x)  # --dropout
        x = residual + x
        x = _normalize(x, self.norm2_kg)

        residual = x
        x = self.encoder_attention(
            query=x,
            key=encoder_output,
            value=encoder_output,
            mask=encoder_mask
        )
        x = self.dropout(x)  # --dropout
        x = residual + x
        x = _normalize(x, self.norm2)

        # finally the ffn
        residual = x
        x = self.ffn(x)
        x = self.dropout(x)  # --dropout
        x = residual + x
        x = _normalize(x, self.norm3)

        return x


class TransformerDecoderKG(nn.Module):
    """
    Transformer Decoder layer.

    :param int n_heads: the number of multihead attention heads.
    :param int n_layers: number of transformer layers.
    :param int embedding_size: the embedding sizes. Must be a multiple of n_heads.
    :param int ffn_size: the size of the hidden layer in the FFN
    :param embedding: an embedding matrix for the bottom layer of the transformer.
        If none, one is created for this encoder.
    :param float dropout: Dropout used around embeddings and before layer
        layer normalizations. This is used in Vaswani 2017 and works well on
        large datasets.
    :param float attention_dropout: Dropout performed after the multhead attention
        softmax. This is not used in Vaswani 2017.
    :param float relu_dropout: Dropout used after the ReLU in the FFN. Not used
        in Vaswani 2017, but used in Tensor2Tensor.
    :param int padding_idx: Reserved padding index in the embeddings matrix.
    :param bool learn_positional_embeddings: If off, sinusoidal embeddings are
        used. If on, position embeddings are learned from scratch.
    :param bool embeddings_scale: Scale embeddings relative to their dimensionality.
        Found useful in fairseq.
    :param int n_positions: Size of the position embeddings matrix.
    """

    def __init__(
            self,
            n_heads,
            n_layers,
            embedding_size,
            ffn_size,
            vocabulary_size,
            embedding,
            dropout=0.0,
            attention_dropout=0.0,
            relu_dropout=0.0,
            embeddings_scale=True,
            learn_positional_embeddings=False,
            padding_idx=None,
            n_positions=1024,
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.ffn_size = ffn_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dim = embedding_size
        self.embeddings_scale = embeddings_scale
        self.dropout = nn.Dropout(dropout)  # --dropout

        self.out_dim = embedding_size
        assert embedding_size % n_heads == 0, \
            'Transformer embedding size must be a multiple of n_heads'

        self.embeddings = embedding

        # create the positional embeddings
        self.position_embeddings = nn.Embedding(n_positions, embedding_size)
        if not learn_positional_embeddings:
            create_position_codes(
                n_positions, embedding_size, out=self.position_embeddings.weight
            )
        else:
            nn.init.normal_(self.position_embeddings.weight, 0, embedding_size ** -0.5)

        # build the model
        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(TransformerDecoderLayerKG(
                n_heads, embedding_size, ffn_size,
                attention_dropout=attention_dropout,
                relu_dropout=relu_dropout,
                dropout=dropout,
            ))

    def forward(self, input, encoder_state, kg_encoder_output, kg_encoder_mask,
                db_encoder_output, db_encoder_mask, incr_state=None):
        encoder_output, encoder_mask = encoder_state

        seq_len = input.size(1)
        positions = input.new(seq_len).long()  # (seq_len)
        positions = torch.arange(seq_len, out=positions).unsqueeze(0)  # (1, seq_len)
        tensor = self.embeddings(input)  # (bs, seq_len, embed_dim)
        if self.embeddings_scale:
            tensor = tensor * np.sqrt(self.dim)
        tensor = tensor + self.position_embeddings(positions).expand_as(tensor)
        tensor = self.dropout(tensor)  # --dropout

        for layer in self.layers:
            tensor = layer(tensor, encoder_output, encoder_mask, kg_encoder_output, kg_encoder_mask, db_encoder_output,
                           db_encoder_mask)

        return tensor, None
