# @Time   : 2020/12/4
# @Author : Chenzhan Shang
# @Email  : czshang@outlook.com

# UPDATE
# @Time   : 2020/12/29, 2021/1/4
# @Author : Xiaolei Wang, Yuanhang Zhou
# @email  : wxl1999@foxmail.com, sdzyh002@gmail.com

r"""
ReDial_Conv
===========
References:
    Li, Raymond, et al. `"Towards deep conversational recommendations."`_ in NeurIPS.

.. _`"Towards deep conversational recommendations."`:
   https://papers.nips.cc/paper/2018/hash/800de15c79c8d840f4e78d3af937d4d4-Abstract.html

"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from crslab.model.base import BaseModel
from crslab.model.utils.functions import sort_for_packed_sequence
from crslab.utils import ModelType


class ReDialModel(BaseModel):
    """

    Attributes:
        vocab_size: A integer indicating the vocabulary size.
        pad_token_idx: A integer indicating the id of padding token.
        start_token_idx: A integer indicating the id of start token.
        end_token_idx: A integer indicating the id of end token.
        unk_token_idx: A integer indicating the id of unk token.
        pretrained_embedding: A string indicating the path of pretrained embedding.
        embedding_dim: A integer indicating the dimension of item embedding.
        utterance_encoder_hidden_size: A integer indicating the size of hidden size in utterance encoder.
        dialog_encoder_hidden_size: A integer indicating the size of hidden size in dialog encoder.
        dialog_encoder_num_layers: A integer indicating the number of layers in dialog encoder.
        use_dropout: A boolean indicating if we use the dropout.
        dropout: A float indicating the dropout rate.
        decoder_hidden_size: A integer indicating the size of hidden size in decoder.
        decoder_num_layers: A integer indicating the number of layer in decoder.
        decoder_embedding_dim: A integer indicating the dimension of embedding in decoder.

    """
    model_type = ModelType.GENERATION

    def __init__(self, opt, device, other_data):
        """

        Args:
            opt (dict): A dictionary record the hyper parameters.
            device (torch.device): A variable indicating which device to place the data and model.
            other_data (dict): A dictionary record the other data.

        """
        # dataset
        self.vocab_size = other_data['vocab']['vocab_size']
        self.pad_token_idx = other_data['vocab']['pad']
        self.start_token_idx = other_data['vocab']['start']
        self.end_token_idx = other_data['vocab']['end']
        self.unk_token_idx = other_data['vocab']['unk']
        self.pretrained_embedding = other_data.get('embedding', None)
        self.embedding_dim = opt.get('embedding_dim', None)
        if opt.get('embedding', None) and self.embedding_dim is None:
            raise
        # AutoRec
        self.n_entity = other_data['vocab']['n_entity']
        self.layer_sizes = opt['autorec_layer_sizes']
        self.pad_entity_idx = other_data['vocab']['pad_entity']
        # HRNN
        self.utterance_encoder_hidden_size = opt['utterance_encoder_hidden_size']
        self.dialog_encoder_hidden_size = opt['dialog_encoder_hidden_size']
        self.dialog_encoder_num_layers = opt['dialog_encoder_num_layers']
        self.use_dropout = opt['use_dropout']
        self.dropout = opt['dropout']
        # SwitchingDecoder
        self.decoder_hidden_size = opt['decoder_hidden_size']
        self.decoder_num_layers = opt['decoder_num_layers']
        self.decoder_embedding_dim = opt['decoder_embedding_dim']

        super(ReDialModel, self).__init__(opt, device)

    def build_model(self):
        self.autorec = AutoRec(self.opt, self.n_entity, self.layer_sizes, self.pad_entity_idx)

        if self.opt.get('embedding', None) and self.pretrained_embedding is not None:
            embedding = nn.Embedding.from_pretrained(
                torch.as_tensor(self.pretrained_embedding, dtype=torch.float), freeze=False,
                padding_idx=self.pad_token_idx)
        else:
            embedding = nn.Embedding(self.vocab_size, self.embedding_dim)

        self.encoder = HRNN(
            embedding=embedding,
            utterance_encoder_hidden_size=self.utterance_encoder_hidden_size,
            dialog_encoder_hidden_size=self.dialog_encoder_hidden_size,
            dialog_encoder_num_layers=self.dialog_encoder_num_layers,
            use_dropout=self.use_dropout,
            dropout=self.dropout,
            pad_token_idx=self.pad_token_idx
        )

        self.decoder = SwitchingDecoder(
            hidden_size=self.decoder_hidden_size,
            context_size=self.dialog_encoder_hidden_size,
            num_layers=self.decoder_num_layers,
            vocab_size=self.vocab_size,
            embedding=embedding,
            pad_token_idx=self.pad_token_idx
        )
        self.loss = nn.CrossEntropyLoss(ignore_index=self.pad_token_idx)

    def recommend(self, batch, mode):
        assert mode in ('train', 'valid', 'test')
        if mode == 'train':
            self.train()
        else:
            self.eval()
        return self.autorec(batch)

    def converse(self, batch, mode):
        """
        Args:
            batch: ::

                {
                    'context': (batch_size, max_context_length, max_utterance_length),
                    'context_lengths': (batch_size),
                    'utterance_lengths': (batch_size, max_context_length),
                    'request': (batch_size, max_utterance_length),
                    'request_lengths': (batch_size),
                    'response': (batch_size, max_utterance_length)
                }
            mode: str

        """
        assert mode in ('train', 'valid', 'test')
        if mode == 'train':
            self.train()
        else:
            self.eval()

        context = batch['context']
        utterance_lengths = batch['utterance_lengths']
        context_lengths = batch['context_lengths']
        context_state = self.encoder(context, utterance_lengths,
                                     context_lengths)  # (batch_size, context_encoder_hidden_size)

        request = batch['request']
        request_lengths = batch['request_lengths']
        log_probs = self.decoder(request, request_lengths,
                                 context_state)  # (batch_size, max_utterance_length, vocab_size + 1)
        preds = log_probs.argmax(dim=-1)  # (batch_size, max_utterance_length)

        log_probs = log_probs.view(-1, log_probs.shape[-1])
        response = batch['response'].view(-1)
        loss = self.loss(log_probs, response)

        return loss, preds

    def forward(self, batch, stage, mode):
        if stage == "rec":
            return self.recommend(batch, mode)
        elif stage == "conv":
            return self.converse(batch, mode)


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


class AutoRec(nn.Module):
    """

    Attributes:
        n_entity: A integer indicating the number of entities.
        layer_sizes: A integer indicating the size of layer in autorec.
        pad_entity_idx: A integer indicating the id of entity padding.

    """

    def __init__(self, opt, n_entity, layer_sizes, pad_entity):
        super(AutoRec, self).__init__()
        self.opt = opt
        self.n_entity = n_entity
        self.layer_sizes = layer_sizes
        self.pad_entity_idx = pad_entity

        # AutoRec
        if self.opt['autorec_f'] == 'identity':
            self.f = lambda x: x
        elif self.opt['autorec_f'] == 'sigmoid':
            self.f = nn.Sigmoid()
        elif self.opt['autorec_f'] == 'relu':
            self.f = nn.ReLU()
        else:
            raise ValueError("Got invalid function name for f : {}".format(self.opt['autorec_f']))

        if self.opt['autorec_g'] == 'identity':
            self.g = lambda x: x
        elif self.opt['autorec_g'] == 'sigmoid':
            self.g = nn.Sigmoid()
        elif self.opt['autorec_g'] == 'relu':
            self.g = nn.ReLU()
        else:
            raise ValueError("Got invalid function name for g : {}".format(self.opt['autorec_g']))

        self.encoder = nn.ModuleList([nn.Linear(self.n_entity, self.layer_sizes[0]) if i == 0
                                      else nn.Linear(self.layer_sizes[i - 1], self.layer_sizes[i])
                                      for i in range(len(self.layer_sizes))])
        self.user_repr_dim = self.layer_sizes[-1]
        self.decoder = nn.Linear(self.user_repr_dim, self.n_entity)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, batch):
        """

        Args:
            batch: ::

                {
                    'context_entities': (batch_size, n_entity),
                    'item': (batch_size)
                }

        """
        context_entities = batch['context_entities']
        for i, layer in enumerate(self.encoder):
            context_entities = self.f(layer(context_entities))
        scores = self.g(self.decoder(context_entities))
        loss = self.loss(scores, batch['item'])

        return loss, scores
