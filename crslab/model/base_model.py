# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/24, 2020/11/27
# @Author : Kun Zhou, Xiaolei Wang
# @Email  : francis_kun_zhou@163.com, wxl1999@foxmail.com

from abc import ABC, abstractmethod
import numpy as np
import torch
from torch import nn


class BaseModel(ABC, nn.Module):
    r"""Base class for all models
    """

    def __init__(self, config, device):
        super(BaseModel, self).__init__()
        self.config = config
        self.device = device

    @abstractmethod
    def build_model(self):
        raise NotImplementedError

    @staticmethod
    def init_embedding(vocab_size, dim, pad_idx, init=None):
        embedding = nn.Embedding(vocab_size, dim, pad_idx)
        nn.init.normal_(embedding.weight, 0, dim ** -0.5)
        nn.init.constant_(embedding.weight[pad_idx], 0)
        if init:
            try:
                embedding.weight.data.copy_(torch.from_numpy(np.load(init)))
            except Exception:
                raise FileNotFoundError
        return embedding

    def forward(self, batch, mode='train'):
        r"""Calculate the training loss for a batch data.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            torch.Tensor: Training loss, shape: []
        """
        pass
