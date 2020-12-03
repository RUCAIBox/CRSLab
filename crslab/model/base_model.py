# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/24, 2020/11/30
# @Author : Kun Zhou, Xiaolei Wang
# @Email  : francis_kun_zhou@163.com, wxl1999@foxmail.com

from abc import ABC, abstractmethod

from torch import nn


class BaseModel(ABC, nn.Module):
    """Base class for all models"""

    def __init__(self, opt, device):
        super(BaseModel, self).__init__()
        self.opt = opt
        self.device = device

    @abstractmethod
    def build_model(self, *args, **kwargs):
        """build model"""
        pass

    def forward(self, batch, mode='train'):
        """calculate loss and prediction for batch under certrain mode

        Args:
            batch (list of dict or tuple): batch data
            mode (str, optional): train/valid/test. Defaults to 'train'.
        """
        pass
