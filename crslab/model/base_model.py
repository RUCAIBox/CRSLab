# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/24
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

#from crslab.model.kgsf_model import KGSFModel
import torch
from torch import nn
import numpy as np


class BaseModel(nn.Module):
    r"""Base class for all models
    """
    def __init__(self, config, device):
        super(BaseModel, self).__init__()
        self.config=config
        self.device=device

    def _build_model(self):
        raise NotImplementedError

    @staticmethod
    def init_embedding(vocab_size, dim, pad_idx, init=None):
        embedding = nn.Embedding(vocab_size, dim, pad_idx)
        nn.init.normal_(embedding.weight, 0, dim ** -0.5)
        nn.init.constant_(embedding.weight[pad_idx], 0)
        if init != None:
            try:
                embedding.weight.data.copy_(torch.from_numpy(np.load(init)))
            except:
                raise FileNotFoundError
        return embedding

    def forward(self, batch, mode='train'):
        r"""Calculate the training loss for a batch data.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            torch.Tensor: Training loss, shape: []
        """
        raise NotImplementedError

    def predict(self, batch):
        r"""Predict the scores between users and items.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            torch.Tensor: Predicted scores for given users and items, shape: [batch_size]
        """
        raise NotImplementedError
