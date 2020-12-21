# @Time   : 2020/12/4
# @Author : Chenzhan Shang
# @Email  : czshang@outlook.com

# UPDATE
# @Time   : 2020/12/18, 2020/12/21
# @Author : Xiaolei Wang, Chenzhan Shang
# @email  : wxl1999@foxmail.com, czshang@outlook.com

import torch
from torch import nn

from crslab.model.base_model import BaseModel
from crslab.model.redial.layers import HRNN, SwitchingDecoder


class ReDialConvModel(BaseModel):
    def __init__(self, opt, device, vocab, side_data):
        # dataset
        self.vocab_size = vocab['vocab_size']
        self.pad_token_idx = vocab['pad']
        self.start_token_idx = vocab['start']
        self.end_token_idx = vocab['end']
        self.unk_token_idx = vocab['unk']
        self.pretrained_embedding = side_data.get('embedding', None)
        self.embedding_dim = opt.get('embedding_dim', None)
        if self.pretrained_embedding is None and self.embedding_dim is None:
            raise
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

        super(ReDialConvModel, self).__init__(opt, device)

    def build_model(self):
        if self.pretrained_embedding:
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

    def forward(self, batch, mode):
        """
        batch: {
            context: (batch_size, max_context_length, max_utterance_length),
            context_lengths: (batch_size),
            utterance_lengths: (batch_size, max_context_length),
            request: (batch_size, max_utterance_length),
            request_lengths: (batch_size),
            response: (batch_size, max_utterance_length)
        }
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

        # Skip null requests
        mask = []
        for i in range(len(request_lengths)):
            if request_lengths[i].item() != 0:
                mask.append(i)
        request = request[mask]
        request_lengths = request_lengths[mask]
        context_state = context_state[mask]
        response = batch['response'][mask]

        log_probs = self.decoder(request, request_lengths,
                                 context_state)  # (batch_size, max_utterance_length, vocab_size + 1)
        preds = log_probs.argmax(dim=-1)  # (batch_size, max_utterance_length)

        log_probs = log_probs.view(-1, log_probs.shape[-1])
        response = response.view(-1)
        loss = self.loss(log_probs, response)

        return loss, preds
