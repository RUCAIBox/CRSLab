# @Time   : 2020/12/4
# @Author : Chenzhan Shang
# @Email  : czshang@outlook.com

# UPDATE:
# @Time   : 2020/12/16
# @Author : Xiaolei Wang
# @Email  : wxl1999@foxmail.com

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from crslab.model.utils.functions import sort_for_packed_sequence


class HRNN(nn.Module):
    def __init__(self,
                 utterance_encoder_hidden_size,
                 dialog_encoder_hidden_size,
                 dialog_encoder_num_layers,
                 pad_token_idx,
                 embedding=None,
                 use_dropout=False,
                 dropout=0.3):
        super(HRNN, self).__init__()
        self.pad_token_idx = pad_token_idx
        # embedding
        self.embedding_size = embedding.weight.shape[1]
        self.embedding = embedding
        # utterance encoder
        self.utterance_encoder_hidden_size = utterance_encoder_hidden_size
        self.utterance_encoder = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.utterance_encoder_hidden_size,
            batch_first=True,
            bidirectional=True
        )
        # conversation encoder
        self.dialog_encoder = nn.GRU(
            input_size=(2 * self.utterance_encoder_hidden_size),
            hidden_size=dialog_encoder_hidden_size,
            num_layers=dialog_encoder_num_layers,
            batch_first=True
        )
        # dropout
        self.use_dropout = use_dropout
        if self.use_dropout:
            self.dropout = nn.Dropout(p=dropout)

    def get_utterance_encoding(self, context, utterance_lengths):
        """
        :param context: (batch_size, max_conversation_length, max_utterance_length)
        :param utterance_lengths: (batch_size, max_conversation_length)
        :return utterance_encoding: (batch_size, max_conversation_length, 2 * utterance_encoder_hidden_size)
        """
        batch_size, max_conv_length = context.shape[:2]
        utterance_lengths = utterance_lengths.reshape(-1)  # (bs * conv_len)
        sorted_lengths, sorted_idx, rev_idx = sort_for_packed_sequence(utterance_lengths)

        # reshape and reorder
        sorted_utterances = context.view(batch_size * max_conv_length, -1).index_select(0, sorted_idx)

        # consider valid sequences only(length > 0)
        num_positive_lengths = torch.sum(utterance_lengths > 0)
        sorted_utterances = sorted_utterances[:num_positive_lengths]
        sorted_lengths = sorted_lengths[:num_positive_lengths]

        embedded = self.embedding(sorted_utterances)
        if self.use_dropout:
            embedded = self.dropout(embedded)

        packed_utterances = pack_padded_sequence(embedded, sorted_lengths, batch_first=True)
        _, utterance_encoding = self.utterance_encoder(packed_utterances)

        # concat the hidden states of the last layer (two directions of the GRU)
        utterance_encoding = torch.cat((utterance_encoding[-1], utterance_encoding[-2]), 1)
        if self.use_dropout:
            utterance_encoding = self.dropout(utterance_encoding)

        # complete the missing sequences (of length 0)
        if num_positive_lengths < batch_size * max_conv_length:
            pad_tensor = utterance_encoding.new_full(
                (batch_size * max_conv_length - num_positive_lengths, 2 * self.utterance_encoder_hidden_size),
                self.pad_token_idx)
            utterance_encoding = torch.cat((utterance_encoding, pad_tensor), 0)

        # retrieve original utterance order and Reshape to separate contexts
        utterance_encoding = utterance_encoding.index_select(0, rev_idx)
        utterance_encoding = utterance_encoding.view(batch_size, max_conv_length,
                                                     2 * self.utterance_encoder_hidden_size)
        return utterance_encoding

    def forward(self, context, utterance_lengths, dialog_lengths):
        """
        :param context: (batch_size, max_context_length, max_utterance_length)
        :param utterance_lengths: (batch_size, max_context_length)
        :param dialog_lengths: (batch_size)
        :return context_state: (batch_size, context_encoder_hidden_size)
        """
        utterance_encoding = self.get_utterance_encoding(context, utterance_lengths)  # (bs, conv_len, 2 * utt_dim)
        sorted_lengths, sorted_idx, rev_idx = sort_for_packed_sequence(dialog_lengths)

        # reorder in decreasing sequence length
        sorted_representations = utterance_encoding.index_select(0, sorted_idx)
        packed_sequences = pack_padded_sequence(sorted_representations, sorted_lengths, batch_first=True)

        _, context_state = self.dialog_encoder(packed_sequences)
        context_state = context_state.index_select(1, rev_idx)
        if self.use_dropout:
            context_state = self.dropout(context_state)
        return context_state[-1]


class SwitchingDecoder(nn.Module):
    def __init__(self, hidden_size, context_size, num_layers, vocab_size, embedding, pad_token_idx):
        super(SwitchingDecoder, self).__init__()
        self.pad_token_idx = pad_token_idx
        self.hidden_size = hidden_size
        self.context_size = context_size
        self.num_layers = num_layers
        if context_size != hidden_size:
            raise ValueError("The context size {} must match the hidden size {} in DecoderGRU".format(
                context_size, hidden_size))

        self.embedding = embedding
        embedding_dim = embedding.weight.shape[1]
        self.decoder = nn.GRU(input_size=embedding_dim, hidden_size=hidden_size,
                              num_layers=num_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, vocab_size)
        self.switch = nn.Linear(hidden_size + context_size, 1)

    def forward(self, request, request_lengths, context_state):
        """
        :param request: (batch_size, max_utterance_length)
        :param request_lengths: (batch_size)
        :param context_state: (batch_size, context_encoder_hidden_size)
        :return log_probabilities: (batch_size, max_utterance_length, vocab_size + 1)
        """
        batch_size, max_utterance_length = request.shape

        # sort for pack
        sorted_lengths, sorted_idx, rev_idx = sort_for_packed_sequence(request_lengths)
        sorted_request = request.index_select(0, sorted_idx)
        embedded_request = self.embedding(sorted_request)  # (batch_size, max_utterance_length, embed_dim)
        packed_request = pack_padded_sequence(embedded_request, sorted_lengths, batch_first=True)

        sorted_context_state = context_state.index_select(0, sorted_idx)
        h_0 = sorted_context_state.unsqueeze(0).expand(
            self.num_layers, batch_size, self.hidden_size
        ).contiguous()  # require context_size == hidden_size

        sorted_vocab_state, _ = self.decoder(packed_request, h_0)
        sorted_vocab_state, _ = pad_packed_sequence(sorted_vocab_state,
                                                    batch_first=True)  # (batch_size, max_request_length, decoder_hidden_size)

        _, max_request_length, decoder_hidden_size = sorted_vocab_state.shape
        pad_tensor = sorted_vocab_state.new_full(
            (batch_size, max_utterance_length - max_request_length, decoder_hidden_size), self.pad_token_idx)
        sorted_vocab_state = torch.cat((sorted_vocab_state, pad_tensor),
                                       dim=1)  # (batch_size, max_utterance_length, decoder_hidden_size)
        sorted_language_output = self.out(sorted_vocab_state)  # (batch_size, max_utterance_length, vocab_size)

        # expand context to each time step
        expanded_sorted_context_state = sorted_context_state.unsqueeze(1).expand(
            batch_size, max_utterance_length, self.context_size
        ).contiguous()  # (batch_size, max_utterance_length, context_size)
        # compute switch
        switch_input = torch.cat((expanded_sorted_context_state, sorted_vocab_state),
                                 dim=2)  # (batch_size, max_utterance_length, context_size + decoder_hidden_size)
        switch = self.switch(switch_input)  # (batch_size, max_utterance_length, 1)

        sorted_output = torch.cat((
            F.logsigmoid(switch) + F.log_softmax(sorted_language_output, dim=2),
            F.logsigmoid(-switch)  # for item
        ), dim=2)
        output = sorted_output.index_select(0, rev_idx)  # (batch_size, max_utterance_length, vocab_size + 1)

        return output
