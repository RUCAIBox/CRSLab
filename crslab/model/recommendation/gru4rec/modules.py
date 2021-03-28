import torch
from torch import nn


class Embedding(nn.Module):
    def __init__(self, item_size, embedding_dim):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(item_size, embedding_dim)

    def forward(self, input: torch.Tensor):
        return self.embedding(input)


class GRU4REC(nn.Module):
    def __init__(self, item_size, embedding_dim, hidden_size, num_layers, dropout_hidden):
        super(GRU4REC, self).__init__()
        self.module_dict = nn.ModuleDict({
            'gru': nn.GRU(embedding_dim,
                          hidden_size,
                          num_layers,
                          dropout=dropout_hidden,
                          batch_first=True),
            'item_embeddings': Embedding(item_size, embedding_dim),
        })
        # self.param = nn.ParameterDict({
        #     'hidden_size': hidden_size
        # })
        self.hidden_size = hidden_size
        # self.item_embeddings = Embedding(item_size, embedding_dim)
        # self.gru = nn.GRU(embedding_dim,
        #                   hidden_size,
        #                   num_layers,
        #                   dropout=dropout_hidden,
        #                   batch_first=True)
        # self.rec_loss = self.cross_entropy

    def cross_entropy(self, seq_out, pos_ids, neg_ids):
        # [batch seq_len hidden_size]
        pos_emb = self.module_dict['item_embeddings'](pos_ids)
        neg_emb = self.module_dict['item_embeddings'](neg_ids)

        # [batch*seq_len hidden_size]
        pos = pos_emb.view(-1, pos_emb.size(2))
        neg = neg_emb.view(-1, neg_emb.size(2))

        # [batch*seq_len hidden_size]
        seq_emb = seq_out.contiguous().view(-1, self.hidden_size)

        # [batch*seq_len]
        pos_logits = torch.sum(pos * seq_emb, -1)
        neg_logits = torch.sum(neg * seq_emb, -1)

        # [batch*seq_len]
        istarget = (pos_ids > 0).view(pos_ids.size(0) * pos_ids.size(1)).float()
        loss = torch.sum(
            - torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget -
            torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        ) / torch.sum(istarget)

        return loss

    def forward(self, input: torch.Tensor):
        return self.module_dict['gru'](input)
