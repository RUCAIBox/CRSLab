# @Time   : 2020/12/16
# @Author : Yuanhang Zhou
# @Email  : sdzyh002@gmail.com

# UPDATE
# @Time   : 2020/12/29, 2021/1/4
# @Author : Xiaolei Wang, Yuanhang Zhou
# @email  : wxl1999@foxmail.com, sdzyh002@gmail.com

r"""
GRU4REC
=======
References:
    Hidasi, Balázs, et al. `"Session-Based Recommendations with Recurrent Neural Networks."`_ in ICLR 2016.

.. _`"Session-Based Recommendations with Recurrent Neural Networks."`:
   https://arxiv.org/abs/1511.06939

"""

import torch
from loguru import logger
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from crslab.model.base import BaseModel


class GRU4RECModel(BaseModel):
    """

    Attributes:
        item_size: A integer indicating the number of items.
        hidden_size: A integer indicating the hidden state size in GRU.
        num_layers: A integer indicating the number of GRU layers.
        dropout_hidden: A float indicating the dropout rate to dropout hidden state.
        dropout_input: A integer indicating if we dropout the input of model.
        embedding_dim: A integer indicating the dimension of item embedding.
        batch_size: A integer indicating the batch size.

    """

    def __init__(self, opt, device, vocab, side_data):
        """

        Args:
            opt (dict): A dictionary record the hyper parameters.
            device (torch.device): A variable indicating which device to place the data and model.
            vocab (dict): A dictionary record the vocabulary information.
            side_data (dict): A dictionary record the side data.

        """
        self.item_size = vocab['n_entity'] + 1
        self.hidden_size = opt['gru_hidden_size']
        self.num_layers = opt['num_layers']
        self.dropout_hidden = opt['dropout_hidden']
        self.dropout_input = opt['dropout_input']
        self.embedding_dim = opt['embedding_dim']
        self.batch_size = opt['batch_size']

        super(GRU4RECModel, self).__init__(opt, device)

    def build_model(self):
        self.item_embeddings = nn.Embedding(self.item_size, self.embedding_dim)
        self.gru = nn.GRU(self.embedding_dim,
                          self.hidden_size,
                          self.num_layers,
                          dropout=self.dropout_hidden,
                          batch_first=True)

        logger.debug('[Finish build rec layer]')

    def reconstruct_input(self, input_ids):
        """
        convert the padding from left to right
        """

        def reverse_padding(ids):
            ans = [0] * len(ids)
            idx = 0
            for m_id in ids:
                m_id = m_id.item()
                if m_id != 0:
                    ans[idx] = m_id
                    idx += 1
            return ans

        input_len = [torch.sum((ids != 0).long()).item() for ids in input_ids]
        input_ids = [reverse_padding(ids) for ids in input_ids]
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_mask = (input_ids != 0).long()

        return input_ids.to(self.device), input_len, input_mask.to(self.device)

    def cross_entropy(self, seq_out, pos_ids, neg_ids, input_mask):
        # [batch seq_len hidden_size]
        pos_emb = self.item_embeddings(pos_ids)
        neg_emb = self.item_embeddings(neg_ids)

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

    def forward(self, batch, mode):
        """
        Args:
            input_ids: padding in left, [pad, pad, id1, id2, ..., idn]
            target_ids: padding in left, [pad, pad, id2, id3, ..., y]
        """
        context, mask, input_ids, target_pos, input_mask, sample_negs, y = batch

        input_ids, input_len, input_mask = self.reconstruct_input(input_ids)
        target_pos, _, _ = self.reconstruct_input(target_pos)
        sample_negs, _, _ = self.reconstruct_input(sample_negs)
        embedded = self.item_embeddings(input_ids)  # (batch, seq_len, hidden_size)
        input_len = [len_ if len_ > 0 else 1 for len_ in input_len]
        embedded = pack_padded_sequence(
            embedded, input_len, enforce_sorted=False,
            batch_first=True)  # (num_layers , batch, hidden_size)

        output, hidden = self.gru(embedded)
        output, output_len = pad_packed_sequence(output, batch_first=True)

        batch, seq_len, hidden_size = output.size()
        logit = output.view(batch, seq_len, hidden_size)

        last_logit = logit[:, -1, :]
        rec_scores = torch.matmul(last_logit, self.item_embeddings.weight.data.T)
        rec_scores = rec_scores.squeeze(1)

        max_out_len = max([len_ for len_ in output_len])
        rec_loss = self.cross_entropy(logit, target_pos[:, :max_out_len],
                                      sample_negs[:, :max_out_len], input_mask)

        return rec_loss, rec_scores
