# @Time   : 2020/12/4
# @Author : Chenzhan Shang
# @Email  : czshang@outlook.com

# UPDATE
# @Time   : 2020/12/29, 2021/1/4
# @Author : Xiaolei Wang, Yuanhang Zhou
# @email  : wxl1999@foxmail.com, sdzyh002@gmail.com

r"""
ReDial_Rec
==========
References:
    Li, Raymond, et al. `"Towards deep conversational recommendations."`_ in NeurIPS.

.. _`"Towards deep conversational recommendations."`:
   https://papers.nips.cc/paper/2018/hash/800de15c79c8d840f4e78d3af937d4d4-Abstract.html

"""

import torch.nn as nn

from crslab.model.base import BaseModel


class ReDialRecModel(BaseModel):
    """

    Attributes:
        n_entity: A integer indicating the number of entities.
        layer_sizes: A integer indicating the size of layer in autorec.
        pad_entity_idx: A integer indicating the id of entity padding.

    """

    def __init__(self, opt, device, vocab, side_data):
        """

        Args:
            opt (dict): A dictionary record the hyper parameters.
            device (torch.device): A variable indicating which device to place the data and model.
            vocab (dict): A dictionary record the vocabulary information.
            side_data (dict): A dictionary record the side data.

        """
        self.n_entity = vocab['n_entity']
        self.layer_sizes = opt['autorec_layer_sizes']
        self.pad_entity_idx = vocab['pad_entity']

        super(ReDialRecModel, self).__init__(opt, device)

    def build_model(self):
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

    def forward(self, batch, mode):
        """

        Args:
            batch: ::

                {
                    'context_entities': (batch_size, n_entity),
                    'item': (batch_size)
                }

            mode (str)

        """
        context_entities = batch['context_entities']
        for i, layer in enumerate(self.encoder):
            context_entities = self.f(layer(context_entities))
        scores = self.g(self.decoder(context_entities))
        loss = self.loss(scores, batch['item'])

        return loss, scores
