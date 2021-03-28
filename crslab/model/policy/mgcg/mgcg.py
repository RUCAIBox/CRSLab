# @Time   : 2020/12/17
# @Author : Yuanhang Zhou
# @Email  : sdzyh002@gmail

# UPDATE
# @Time   : 2020/12/29, 2021/1/4
# @Author : Xiaolei Wang, Yuanhang Zhou
# @email  : wxl1999@foxmail.com, sdzyh002@gmail.com

r"""
MGCG
====
References:
    Liu, Zeming, et al. `"Towards Conversational Recommendation over Multi-Type Dialogs."`_ in ACL 2020.

.. _"Towards Conversational Recommendation over Multi-Type Dialogs.":
   https://www.aclweb.org/anthology/2020.acl-main.98/

"""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from crslab.model.base import BaseModel


class MGCGModel(BaseModel):
    """

    Attributes:
        topic_class_num: A integer indicating the number of topic.
        vocab_size: A integer indicating the size of vocabulary.
        embedding_dim: A integer indicating the dimension of embedding layer.
        hidden_size: A integer indicating the size of hidden state.
        num_layers: A integer indicating the number of layers in GRU.
        dropout_hidden: A float indicating the dropout rate of hidden state.
        n_sent: A integer indicating sequence length in user profile.

    """

    def __init__(self, opt, device, vocab, side_data):
        """

        Args:
            opt (dict): A dictionary record the hyper parameters.
            device (torch.device): A variable indicating which device to place the data and model.
            vocab (dict): A dictionary record the vocabulary information.
            side_data (dict): A dictionary record the side data.
        
        """
        self.topic_class_num = vocab['n_topic']
        self.vocab_size = vocab['vocab_size']
        self.embedding_dim = opt['embedding_dim']
        self.hidden_size = opt['hidden_size']
        self.num_layers = opt['num_layers']
        self.dropout_hidden = opt['dropout_hidden']
        self.n_sent = opt.get('n_sent', 10)

        super(MGCGModel, self).__init__(opt, device)

    def build_model(self, *args, **kwargs):
        """build model"""
        self.embeddings = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.context_lstm = nn.LSTM(self.embedding_dim,
                                    self.hidden_size,
                                    self.num_layers,
                                    dropout=self.dropout_hidden,
                                    batch_first=True)

        self.topic_lstm = nn.LSTM(self.embedding_dim,
                                  self.hidden_size,
                                  self.num_layers,
                                  dropout=self.dropout_hidden,
                                  batch_first=True)

        self.profile_lstm = nn.LSTM(self.embedding_dim,
                                    self.hidden_size,
                                    self.num_layers,
                                    dropout=self.dropout_hidden,
                                    batch_first=True)

        self.state2topic_id = nn.Linear(self.hidden_size * 3,
                                        self.topic_class_num)
        self.loss = nn.CrossEntropyLoss()

    def get_length(self, input):
        return [torch.sum((ids != 0).long()).item() for ids in input]

    def forward(self, batch, mode):
        # conv_id, message_id, context, context_mask, topic_path_kw, tp_mask, user_profile, profile_mask, y = batch
        context, context_mask, topic_path_kw, tp_mask, user_profile, profile_mask, y = batch

        len_context = self.get_length(context)
        len_tp = self.get_length(topic_path_kw)
        len_profile = self.get_length(user_profile)

        bs_, word_num = user_profile.shape
        bs = bs_ // self.n_sent

        context = self.embeddings(context)
        topic_path_kw = self.embeddings(topic_path_kw)
        user_profile = self.embeddings(user_profile)

        context = pack_padded_sequence(context,
                                       len_context,
                                       enforce_sorted=False,
                                       batch_first=True)
        topic_path_kw = pack_padded_sequence(topic_path_kw,
                                             len_tp,
                                             enforce_sorted=False,
                                             batch_first=True)
        user_profile = pack_padded_sequence(user_profile,
                                            len_profile,
                                            enforce_sorted=False,
                                            batch_first=True)

        init_h0 = (torch.zeros(self.num_layers, bs,
                               self.hidden_size).to(self.device),
                   torch.zeros(self.num_layers, bs,
                               self.hidden_size).to(self.device))

        # batch, seq_len, num_directions * hidden_size        # num_layers * num_directions, batch, hidden_size
        context_output, (context_h, _) = self.context_lstm(context, init_h0)
        topic_output, (topic_h, _) = self.topic_lstm(topic_path_kw, init_h0)
        # batch*sent_num, seq_len, num_directions * hidden_size
        init_h0 = (torch.zeros(self.num_layers, bs * self.n_sent,
                               self.hidden_size).to(self.device),
                   torch.zeros(self.num_layers, bs * self.n_sent,
                               self.hidden_size).to(self.device))
        profile_output, (profile_h,
                         _) = self.profile_lstm(user_profile, init_h0)

        # batch, hidden_size
        context_rep = context_h[-1]
        topic_rep = topic_h[-1]

        profile_rep = profile_h[-1]
        profile_rep = profile_rep.view(bs, self.n_sent, -1)
        # batch, hidden_size
        profile_rep = torch.mean(profile_rep, dim=1)

        state_rep = torch.cat((context_rep, topic_rep, profile_rep), 1)

        topic_scores = self.state2topic_id(state_rep)

        topic_loss = self.loss(topic_scores, y)

        return topic_loss, topic_scores
