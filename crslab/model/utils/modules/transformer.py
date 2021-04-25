# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/24
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

"""Near infinity, useful as a large penalty for scoring when inf is bad."""
NEAR_INF = 1e20
NEAR_INF_FP16 = 65504


def neginf(dtype):
    """Returns a representable finite number near -inf for a dtype."""
    if dtype is torch.float16:
        return -NEAR_INF_FP16
    else:
        return -NEAR_INF


def _create_selfattn_mask(x):
    # figure out how many timestamps we need
    bsz = x.size(0)
    time = x.size(1)
    # make sure that we don't look into the future
    mask = torch.tril(x.new(time, time).fill_(1))
    # broadcast across batch
    mask = mask.unsqueeze(0).expand(bsz, -1, -1)
    return mask


def create_position_codes(n_pos, dim, out):
    position_enc = np.array([
        [pos / np.power(10000, 2 * j / dim) for j in range(dim // 2)]
        for pos in range(n_pos)
    ])

    out.data[:, 0::2] = torch.as_tensor(np.sin(position_enc))
    out.data[:, 1::2] = torch.as_tensor(np.cos(position_enc))
    out.detach_()
    out.requires_grad = False


def _normalize(tensor, norm_layer):
    """Broadcast layer norm"""
    size = tensor.size()
    return norm_layer(tensor.view(-1, size[-1])).view(size)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, dim, dropout=.0):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.dim = dim

        self.attn_dropout = nn.Dropout(p=dropout)  # --attention-dropout
        self.q_lin = nn.Linear(dim, dim)
        self.k_lin = nn.Linear(dim, dim)
        self.v_lin = nn.Linear(dim, dim)
        # TODO: merge for the initialization step
        nn.init.xavier_normal_(self.q_lin.weight)
        nn.init.xavier_normal_(self.k_lin.weight)
        nn.init.xavier_normal_(self.v_lin.weight)
        # and set biases to 0
        self.out_lin = nn.Linear(dim, dim)

        nn.init.xavier_normal_(self.out_lin.weight)

    def forward(self, query, key=None, value=None, mask=None):
        # Input is [B, query_len, dim]
        # Mask is [B, key_len] (selfattn) or [B, key_len, key_len] (enc attn)
        batch_size, query_len, dim = query.size()
        assert dim == self.dim, \
            f'Dimensions do not match: {dim} query vs {self.dim} configured'
        assert mask is not None, 'Mask is None, please specify a mask'
        n_heads = self.n_heads
        dim_per_head = dim // n_heads
        scale = math.sqrt(dim_per_head)

        def prepare_head(tensor):
            # input is [batch_size, seq_len, n_heads * dim_per_head]
            # output is [batch_size * n_heads, seq_len, dim_per_head]
            bsz, seq_len, _ = tensor.size()
            tensor = tensor.view(batch_size, tensor.size(1), n_heads, dim_per_head)
            tensor = tensor.transpose(1, 2).contiguous().view(
                batch_size * n_heads,
                seq_len,
                dim_per_head
            )
            return tensor

        # q, k, v are the transformed values
        if key is None and value is None:
            # self attention
            key = value = query
        elif value is None:
            # key and value are the same, but query differs
            # self attention
            value = key
        _, key_len, dim = key.size()

        q = prepare_head(self.q_lin(query))
        k = prepare_head(self.k_lin(key))
        v = prepare_head(self.v_lin(value))

        dot_prod = q.div_(scale).bmm(k.transpose(1, 2))
        # [B * n_heads, query_len, key_len]
        attn_mask = (
            (mask == 0)
                .view(batch_size, 1, -1, key_len)
                .repeat(1, n_heads, 1, 1)
                .expand(batch_size, n_heads, query_len, key_len)
                .view(batch_size * n_heads, query_len, key_len)
        )
        assert attn_mask.shape == dot_prod.shape
        dot_prod.masked_fill_(attn_mask, neginf(dot_prod.dtype))

        attn_weights = F.softmax(dot_prod, dim=-1).type_as(query)
        attn_weights = self.attn_dropout(attn_weights)  # --attention-dropout

        attentioned = attn_weights.bmm(v)
        attentioned = (
            attentioned.type_as(query)
                .view(batch_size, n_heads, query_len, dim_per_head)
                .transpose(1, 2).contiguous()
                .view(batch_size, query_len, dim)
        )

        out = self.out_lin(attentioned)

        return out


class TransformerFFN(nn.Module):
    def __init__(self, dim, dim_hidden, relu_dropout=.0):
        super(TransformerFFN, self).__init__()
        self.relu_dropout = nn.Dropout(p=relu_dropout)
        self.lin1 = nn.Linear(dim, dim_hidden)
        self.lin2 = nn.Linear(dim_hidden, dim)
        nn.init.xavier_uniform_(self.lin1.weight)
        nn.init.xavier_uniform_(self.lin2.weight)
        # TODO: initialize biases to 0

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = self.relu_dropout(x)  # --relu-dropout
        x = self.lin2(x)
        return x


class TransformerEncoderLayer(nn.Module):
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
        self.attention = MultiHeadAttention(
            n_heads, embedding_size,
            dropout=attention_dropout,  # --attention-dropout
        )
        self.norm1 = nn.LayerNorm(embedding_size)
        self.ffn = TransformerFFN(embedding_size, ffn_size, relu_dropout=relu_dropout)
        self.norm2 = nn.LayerNorm(embedding_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, tensor, mask):
        tensor = tensor + self.dropout(self.attention(tensor, mask=mask))
        tensor = _normalize(tensor, self.norm1)
        tensor = tensor + self.dropout(self.ffn(tensor))
        tensor = _normalize(tensor, self.norm2)
        tensor *= mask.unsqueeze(-1).type_as(tensor)
        return tensor


class TransformerEncoder(nn.Module):
    """
    Transformer encoder module.

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
    :param bool reduction: If true, returns the mean vector for the entire encoding
        sequence.
    :param int n_positions: Size of the position embeddings matrix.
    """

    def __init__(
            self,
            n_heads,
            n_layers,
            embedding_size,
            ffn_size,
            vocabulary_size,
            embedding=None,
            dropout=0.0,
            attention_dropout=0.0,
            relu_dropout=0.0,
            padding_idx=0,
            learn_positional_embeddings=False,
            embeddings_scale=False,
            reduction=True,
            n_positions=1024
    ):
        super(TransformerEncoder, self).__init__()

        self.embedding_size = embedding_size
        self.ffn_size = ffn_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dim = embedding_size
        self.embeddings_scale = embeddings_scale
        self.reduction = reduction
        self.padding_idx = padding_idx
        # this is --dropout, not --relu-dropout or --attention-dropout
        self.dropout = nn.Dropout(dropout)
        self.out_dim = embedding_size
        assert embedding_size % n_heads == 0, \
            'Transformer embedding size must be a multiple of n_heads'

        # check input formats:
        if embedding is not None:
            assert (
                    embedding_size is None or embedding_size == embedding.weight.shape[1]
            ), "Embedding dim must match the embedding size."

        if embedding is not None:
            self.embeddings = embedding
        else:
            assert False
            assert padding_idx is not None
            self.embeddings = nn.Embedding(
                vocabulary_size, embedding_size, padding_idx=padding_idx
            )
            nn.init.normal_(self.embeddings.weight, 0, embedding_size ** -0.5)

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
            self.layers.append(TransformerEncoderLayer(
                n_heads, embedding_size, ffn_size,
                attention_dropout=attention_dropout,
                relu_dropout=relu_dropout,
                dropout=dropout,
            ))

    def forward(self, input):
        """
            input data is a FloatTensor of shape [batch, seq_len, dim]
            mask is a ByteTensor of shape [batch, seq_len], filled with 1 when
            inside the sequence and 0 outside.
        """
        mask = input != self.padding_idx
        positions = (mask.cumsum(dim=1, dtype=torch.int64) - 1).clamp_(min=0)
        tensor = self.embeddings(input)
        if self.embeddings_scale:
            tensor = tensor * np.sqrt(self.dim)
        tensor = tensor + self.position_embeddings(positions).expand_as(tensor)
        # --dropout on the embeddings
        tensor = self.dropout(tensor)

        tensor *= mask.unsqueeze(-1).type_as(tensor)
        for i in range(self.n_layers):
            tensor = self.layers[i](tensor, mask)

        if self.reduction:
            divisor = mask.type_as(tensor).sum(dim=1).unsqueeze(-1).clamp(min=1e-7)
            output = tensor.sum(dim=1) / divisor
            return output
        else:
            output = tensor
            return output, mask


class TransformerDecoderLayer(nn.Module):
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

        self.ffn = TransformerFFN(embedding_size, ffn_size, relu_dropout=relu_dropout)
        self.norm3 = nn.LayerNorm(embedding_size)

    def forward(self, x, encoder_output, encoder_mask):
        decoder_mask = self._create_selfattn_mask(x)
        # first self attn
        residual = x
        # don't peak into the future!
        x = self.self_attention(query=x, mask=decoder_mask)
        x = self.dropout(x)  # --dropout
        x = x + residual
        x = _normalize(x, self.norm1)

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

    def _create_selfattn_mask(self, x):
        # figure out how many timestamps we need
        bsz = x.size(0)
        time = x.size(1)
        # make sure that we don't look into the future
        mask = torch.tril(x.new(time, time).fill_(1))
        # broadcast across batch
        mask = mask.unsqueeze(0).expand(bsz, -1, -1)
        return mask


class TransformerDecoder(nn.Module):
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
            embedding=None,
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
        self.dropout = nn.Dropout(p=dropout)  # --dropout

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
            self.layers.append(TransformerDecoderLayer(
                n_heads, embedding_size, ffn_size,
                attention_dropout=attention_dropout,
                relu_dropout=relu_dropout,
                dropout=dropout,
            ))

    def forward(self, input, encoder_state, incr_state=None):
        encoder_output, encoder_mask = encoder_state

        seq_len = input.shape[1]
        positions = input.new_empty(seq_len).long()
        positions = torch.arange(seq_len, out=positions).unsqueeze(0)  # (batch, seq_len)
        tensor = self.embeddings(input)
        if self.embeddings_scale:
            tensor = tensor * np.sqrt(self.dim)
        tensor = tensor + self.position_embeddings(positions).expand_as(tensor)
        tensor = self.dropout(tensor)  # --dropout

        for layer in self.layers:
            tensor = layer(tensor, encoder_output, encoder_mask)

        return tensor, None
