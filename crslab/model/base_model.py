# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/24, 2020/11/30
# @Author : Kun Zhou, Xiaolei Wang
# @Email  : francis_kun_zhou@163.com, wxl1999@foxmail.com

from abc import ABC, abstractmethod
import numpy as np
import torch
from torch import nn


def init_embedding(vocab_size, dim, pad_idx=None, pretrained_embedding=None, freeze=True):
    if pretrained_embedding:
        e = nn.Embedding.from_pretrained(pretrained_embedding, freeze, pad_idx)
    else:
        e = nn.Embedding(vocab_size, dim, pad_idx)
    return e


class BaseModel(ABC, nn.Module):
    r"""Base class for all models
    """

    def __init__(self, config, device):
        super(BaseModel, self).__init__()
        self.config = config
        self.device = device

    @abstractmethod
    def build_model(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, batch, mode='train'):
        r"""Calculate the training loss for a batch data.

        Args:

        Returns:
            torch.Tensor: Training loss, shape: []
        """
        pass
