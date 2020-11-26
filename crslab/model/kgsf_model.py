# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/24
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

from crslab.model.base_model import BaseModel
from crslab.model.layers import SelfAttentionSeq, GateLayer, load_copy_mask
from crslab.model.transformer import TransformerEncoder, TransformerDecoderKG
from torch_geometric.nn import GCNConv, RGCNConv
from torch import nn
import os
import torch.nn.functional as F
import torch
from loguru import logger


class KGSFModel(BaseModel):
    r"""
    S3Rec is the first work to incorporate self-supervised learning in
    sequential recommendation.

    NOTE:
        Under this framework, we need reconstruct the pretraining data,
        which would affect the pre-training speed.
    """

    def __init__(self, config, device, side_data):
        super(KGSFModel, self).__init__(config, device)
        self.vocab_size=self.config['vocab_size']
        self.token_emb_size=self.config['token_emb_size']
        self.pad_token_idx=self.config['pad_token_idx']
        self.start_token_idx=self.config['start_token_idx']
        self.end_token_idx=self.config['end_token_idx']

        self.data_path=self.config['data_path']
        self.word2vec_path=os.path.join(self.data_path, self.config['data_files']['word2vec'])
        self.copy_mask_path=os.path.join(self.data_path, self.config['data_files']["copy_mask"])
        self.n_word=self.config['n_word']
        self.n_entity=self.config['n_entity']
        self.kg_emb_size=self.config['kg_emb_size']
        self.word_pad=self.config['word_pad']
        self.entity_pad=self.config['entity_pad']
        self.num_bases=self.config['num_bases']
        self.n_heads = self.config['n_heads']
        self.n_layers = self.config['n_layers']
        self.ffn_size = self.config['ffn_size']
        self.dropout = self.config['dropout']
        self.attention_dropout = self.config['attention_dropout']
        self.relu_dropout = self.config['relu_dropout']
        self.learn_positional_embeddings = self.config['learn_positional_embeddings']
        self.embeddings_scale = self.config['embeddings_scale']
        self.reduction = self.config['reduction']
        self.n_positions = self.config['n_positions']
        self.n_relation = self.config['n_relation']

        entity_edges, word_edges = side_data
        self.entity_edges, self.word_edges=[ele.to(self.device) for ele in entity_edges], word_edges.to(self.device)

        self._build_model()

    def _build_model(self):
        self._init_embeddings()
        self._build_GNN_layer()
        self._build_infomax_layer()
        self._build_recommendation_layer()
        self._build_conversation_layer()

    def _init_embeddings(self):
        #build embeddings, for entity_KGE, the entity_encoder needs not
        self.token_embedding = self.init_embedding(self.vocab_size, self.token_emb_size, self.pad_token_idx, self.word2vec_path)
        self.word_KG_embedding = self.init_embedding(self.n_word, self.kg_emb_size, self.word_pad)
        logger.debug('finish initializing embeddings')

    def _build_GNN_layer(self):
        # db encoder
        self.entity_encoder = RGCNConv(self.n_entity, self.kg_emb_size, self.n_relation, self.num_bases)
        self.entity_self_attn = SelfAttentionSeq(self.kg_emb_size, self.kg_emb_size)

        # concept encoder
        self.word_encoder = GCNConv(self.kg_emb_size, self.kg_emb_size)
        self.word_self_attn = SelfAttentionSeq(self.kg_emb_size, self.kg_emb_size)

        # gate mechanism
        self.gate_layer = GateLayer(self.kg_emb_size)
        logger.debug('finish building GNN layers')

    def _build_infomax_layer(self):
        # infomax requires
        self.infomax_layer = nn.Linear(self.kg_emb_size, self.kg_emb_size)
        self.infomax_bias = nn.Linear(self.kg_emb_size, self.n_entity)
        self.infomax_loss = nn.MSELoss(reduction="none")

    def _build_recommendation_layer(self):
        # recommendation requires
        self.rec_bias = nn.Linear(self.kg_emb_size, self.n_entity)
        self.rec_loss = nn.CrossEntropyLoss(reduction="sum")

    def _build_conversation_layer(self):
        # conversation requires
        self.conv_encoder = TransformerEncoder(
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            embedding_size=self.token_emb_size,
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

        self.conv_entity_norm = nn.Linear(self.kg_emb_size, self.ffn_size)
        self.conv_entity_attn_norm = nn.Linear(self.kg_emb_size, self.ffn_size)
        self.conv_word_norm = nn.Linear(self.kg_emb_size, self.ffn_size)
        self.conv_word_attn_norm = nn.Linear(self.kg_emb_size, self.ffn_size)

        self.copy_norm = nn.Linear(self.ffn_size * 3, self.token_emb_size)
        self.copy_output = nn.Linear(self.token_emb_size, self.vocab_size)
        self.copy_mask = load_copy_mask(self.copy_mask_path)

        self.conv_decoder = TransformerDecoderKG(
            self.n_heads, self.n_layers, self.token_emb_size, self.ffn_size, self.vocab_size,
            embedding=self.token_embedding,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            relu_dropout=self.relu_dropout,
            embeddings_scale=self.embeddings_scale,
            learn_positional_embeddings=self.learn_positional_embeddings,
            padding_idx=self.pad_token_idx,
            n_positions=self.n_positions
        )
        self.conv_loss = nn.CrossEntropyLoss(reduction="none")

    def pretrain_infomax(self, batch):
        '''
        words: batch_size*word_length
        entities: batch_size*vocab_size
        '''
        words, entities=batch
        entity_graph_representations=self.entity_encoder(None, self.entity_edges[0], self.entity_edges[1])
        word_graph_representations=self.word_encoder(self.word_KG_embedding.weight, self.word_edges)

        #entity_padding_mask = entities.eq(self.entity_pad)  # (bs, entity_len)
        entity_loss_mask = torch.sum(entities, -1)
        word_padding_mask = words.eq(self.word_pad)  # (bs, seq_len)

        word_representations=word_graph_representations[words]
        word_attn_rep=self.word_self_attn(word_representations, word_padding_mask)
        word_info_rep = self.infomax_layer(word_attn_rep)  # (bs, dim)
        info_predict = F.linear(word_info_rep, entity_graph_representations, self.infomax_bias.bias)  # (bs, #entity)
        loss = self.infomax_loss(info_predict, entities).sum(-1) * entity_loss_mask
        if torch.sum(entity_loss_mask)==0:
            return torch.sum(loss)
        else:
            loss = torch.sum(loss) / torch.sum(entity_loss_mask)
            return loss

    def recommender(self, batch, mode='train'):
        '''
        context_entities: batch_size*word_length
        context_words: batch_size*entity_length
        movie: batch_size
        '''
        context_entities, context_words, entities, movie = batch
        batch_size=movie.size(0)

        entity_graph_representations = self.entity_encoder(None, self.entity_edges[0], self.entity_edges[1])
        word_graph_representations = self.word_encoder(self.word_KG_embedding.weight, self.word_edges)

        entity_padding_mask = context_entities.eq(self.entity_pad)  # (bs, entity_len)
        word_padding_mask = context_words.eq(self.word_pad)  # (bs, seq_len)
        entity_loss_mask = torch.sum(entities, -1)

        entity_representations = entity_graph_representations[context_entities]
        word_representations = word_graph_representations[context_words]

        entity_attn_rep = self.entity_self_attn(entity_representations, entity_padding_mask)
        word_attn_rep = self.word_self_attn(word_representations, word_padding_mask)

        user_rep = self.gate_layer(entity_attn_rep, word_attn_rep)
        rec_scores = F.linear(user_rep, entity_graph_representations, self.rec_bias.bias)  # (bs, #entity)

        loss = self.rec_loss(rec_scores, movie)/batch_size

        word_info_rep = self.infomax_layer(word_attn_rep)  # (bs, dim)
        info_predict = F.linear(word_info_rep, entity_graph_representations, self.infomax_bias.bias)  # (bs, #entity)
        reg_loss = self.infomax_loss(info_predict, entities).sum(-1) * entity_loss_mask

        if torch.sum(entity_loss_mask)==0:
            reg_loss = torch.sum(reg_loss)
        else:
            reg_loss = torch.sum(reg_loss) / torch.sum(entity_loss_mask)

        return loss, reg_loss, rec_scores

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

        copy_logits = self.copy_output(copy_latent) * self.copy_mask.unsqueeze(0).unsqueeze(0)  # (bs, seq_len, vocab_size)
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
        for _ in range(self.config['response_max_length']):
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
        batch_size = context_tokens.size(0)

        entity_graph_representations = self.entity_encoder(None, self.entity_edges[0], self.entity_edges[1])
        word_graph_representations = self.word_encoder(self.word_KG_embedding.weight, self.word_edges)

        entity_padding_mask = context_entities.eq(self.entity_pad)  # (bs, entity_len)
        word_padding_mask = context_words.eq(self.word_pad)  # (bs, seq_len)

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
        if mode == 'train':
            logits, preds = self._KG_transformer_decode_forced(tokens_encoding, conv_entity_reps, conv_entity_emb, entity_padding_mask,
                                                conv_word_reps, conv_word_emb, word_padding_mask, response)

            logits = logits.view(-1, logits.shape[-1])
            response = response.view(-1)
            response_mask = response.ne(self.pad_token_idx)

            loss = self.conv_loss(logits, response)*response_mask
            return loss.sum()/torch.sum(response_mask)
        else:
            logits, preds = self._KG_transformer_decode_greedy(tokens_encoding, conv_entity_reps, conv_entity_emb, entity_padding_mask,
                                                    conv_word_reps, conv_word_emb, word_padding_mask)
            return preds

    def _starts(self, batch_size):
        """Return bsz start tokens."""
        return torch.tensor([self.start_token_idx], dtype=torch.long).detach().expand(batch_size, 1).cuda()

    def stem_conv_parameters(self):
        for params in [self.token_embedding.parameters(), self.conv_encoder.parameters(),
                  self.conv_entity_norm.parameters(), self.conv_word_norm.parameters(),
                  self.conv_entity_attn_norm.parameters(), self.conv_word_attn_norm.parameters(),
                  self.copy_norm.parameters(), self.copy_output.parameters(),
                  self.conv_decoder.parameters()]:
            for p in params:
                p.requires_grad_(False)

