# @Time   : 2020/12/13
# @Author : Kun Zhou
# @Email  : wxl1999@foxmail.com

# UPDATE
# @Time   : 2020/12/13, 2021/1/4
# @Author : Xiaolei Wang, Yuanhang Zhou
# @email  : wxl1999@foxmail.com, sdzyh002@gmail.com

import copy

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SASRec(nn.Module):
    def __init__(self, hidden_dropout_prob, device, initializer_range,
                 hidden_size, max_seq_length, item_size, num_attention_heads,
                 attention_probs_dropout_prob, hidden_act, num_hidden_layers):
        super(SASRec, self).__init__()
        self.hidden_dropout_prob = hidden_dropout_prob
        self.device = device
        self.initializer_range = initializer_range
        self.hidden_size = hidden_size
        self.max_seq_length = max_seq_length
        self.item_size = item_size
        self.num_attention_heads = num_attention_heads
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.hidden_act = hidden_act
        self.num_hidden_layers = num_hidden_layers

        self.build_model()
        self.init_model()

    def build_model(self):
        self.embeddings = Embeddings(self.item_size, self.hidden_size,
                                     self.max_seq_length,
                                     self.hidden_dropout_prob)
        self.encoder = Encoder(self.num_hidden_layers, self.hidden_size,
                               self.num_attention_heads,
                               self.hidden_dropout_prob, self.hidden_act,
                               self.attention_probs_dropout_prob)

        self.act = nn.Tanh()
        self.dropout = nn.Dropout(p=self.hidden_dropout_prob)

    def init_model(self):
        self.apply(self.init_sas_weights)

    def forward(self,
                input_ids,
                attention_mask=None,
                output_all_encoded_layers=True):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)  # (bs, seq_len)
        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(
            2)  # torch.int64, (bs, 1, 1, seq_len)
        # 添加mask 只关注前几个物品进行推荐
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape),
                                     diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(self.device)
        extended_attention_mask = extended_attention_mask * subsequent_mask

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        # extended_attention_mask = extended_attention_mask.to(
        #   dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        embedding = self.embeddings(input_ids)

        encoded_layers = self.encoder(
            embedding,
            extended_attention_mask,
            output_all_encoded_layers=output_all_encoded_layers)
        # [B L H]
        sequence_output = encoded_layers[-1]
        return sequence_output

    def init_sas_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def save_model(self, file_name):
        torch.save(self.cpu().state_dict(), file_name)
        self.to(self.device)

    def load_model(self, path):
        load_states = torch.load(path, map_location=self.device)
        load_states_keys = set(load_states.keys())
        this_states_keys = set(self.state_dict().keys())
        assert this_states_keys.issubset(this_states_keys)
        key_not_used = load_states_keys - this_states_keys
        for key in key_not_used:
            del load_states[key]

        self.load_state_dict(load_states)

    def compute_loss(self, y_pred, y, subset='test'):
        pass

    def cross_entropy(self, seq_out, pos_ids, neg_ids):
        # [batch seq_len hidden_size]
        pos_emb = self.embeddings.item_embeddings(pos_ids)
        neg_emb = self.embeddings.item_embeddings(neg_ids)

        # [batch*seq_len hidden_size]
        pos = pos_emb.view(-1, pos_emb.size(2))
        neg = neg_emb.view(-1, neg_emb.size(2))

        # [batch*seq_len hidden_size]
        seq_emb = seq_out.view(-1, self.hidden_size)

        # [batch*seq_len]
        pos_logits = torch.sum(pos * seq_emb, -1)
        neg_logits = torch.sum(neg * seq_emb, -1)

        # [batch*seq_len]
        istarget = (pos_ids > 0).view(-1).float()
        loss = torch.sum(-torch.log(torch.sigmoid(pos_logits) + 1e-24) *
                         istarget -
                         torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) *
                         istarget) / torch.sum(istarget)

        return loss


def gelu(x):
    """Implementation of the gelu activation function.

    For information: OpenAI GPT's gelu is slightly different
    (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) *
    (x + 0.044715 * torch.pow(x, 3))))
    Also see https://arxiv.org/abs/1606.08415

    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": F.relu, "swish": swish}


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root)."""
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(hidden_size), requires_grad=True)
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class Embeddings(nn.Module):
    """Construct the embeddings from item, position, attribute."""

    def __init__(self, item_size, hidden_size, max_seq_length,
                 hidden_dropout_prob):
        super(Embeddings, self).__init__()

        self.item_embeddings = nn.Embedding(item_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_seq_length, hidden_size)

        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_ids):
        seq_length = input_ids.size(1)

        position_ids = torch.arange(seq_length,
                                    dtype=torch.long,
                                    device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        items_embeddings = self.item_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = items_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, hidden_dropout_prob,
                 attention_probs_dropout_prob):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(attention_probs_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        """
        Args:
            x: (bs, seq_len, all_head_size)

        Returns:
            x.permute(0, 2, 1, 3), (bs, num_heads, seq_len, head_size)

        """
        new_x_shape = x.size()[:-1] + (self.num_attention_heads,
                                       self.attention_head_size)
        x = x.view(*new_x_shape)

        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(
            -1, -2))  # (bs, num_heads, seq_len, seq_len)

        attention_scores = attention_scores / math.sqrt(
            self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)

        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class Intermediate(nn.Module):
    def __init__(self, hidden_size, hidden_act, hidden_dropout_prob):
        super(Intermediate, self).__init__()
        self.dense_1 = nn.Linear(hidden_size, hidden_size * 4)
        if isinstance(hidden_act, str):
            self.intermediate_act_fn = ACT2FN[hidden_act]
        else:
            self.intermediate_act_fn = hidden_act

        self.dense_2 = nn.Linear(hidden_size * 4, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_tensor):

        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class Layer(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, hidden_dropout_prob,
                 hidden_act, attention_probs_dropout_prob):
        super(Layer, self).__init__()
        self.attention = SelfAttention(hidden_size, num_attention_heads,
                                       hidden_dropout_prob,
                                       attention_probs_dropout_prob)
        self.intermediate = Intermediate(hidden_size, hidden_act, hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        return intermediate_output


class Encoder(nn.Module):
    def __init__(self, num_hidden_layers, hidden_size, num_attention_heads,
                 hidden_dropout_prob, hidden_act,
                 attention_probs_dropout_prob):
        super(Encoder, self).__init__()
        layer = Layer(hidden_size, num_attention_heads, hidden_dropout_prob,
                      hidden_act, attention_probs_dropout_prob)
        self.layer = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(num_hidden_layers)])

    def forward(self,
                hidden_states,
                attention_mask,
                output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers
