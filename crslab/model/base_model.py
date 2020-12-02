# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/24, 2020/11/30
# @Author : Kun Zhou, Xiaolei Wang
# @Email  : francis_kun_zhou@163.com, wxl1999@foxmail.com

from abc import ABC, abstractmethod
import torch
from loguru import logger
from torch import nn


class BaseModel(ABC, nn.Module):
    r"""Base class for all models
    """

    def __init__(self, opt, device):
        super(BaseModel, self).__init__()
        self.opt = opt
        self.device = device

    @abstractmethod
    def build_model(self, *args, **kwargs):
        logger.info(f"[Build model {self.opt['model_name']}]")

    def forward(self, batch, mode='train'):
        r"""Calculate the training loss for a batch data.

        Args:

        Returns:
            torch.Tensor: Training loss, shape: []
        """
        pass
