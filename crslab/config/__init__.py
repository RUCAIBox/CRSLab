# -*- encoding: utf-8 -*-
# @Time    :   2020/12/22
# @Author  :   Xiaolei Wang
# @email   :   wxl1999@foxmail.com

# UPDATE
# @Time    :   2020/12/29
# @Author  :   Xiaolei Wang
# @email   :   wxl1999@foxmail.com

"""Config module which loads parameters for the whole system.

Attributes:
    SAVE_PATH (str): where system to save.
    DATASET_PATH (str): where dataset to save.
    MODEL_PATH (str): where model related data to save.
    PRETRAIN_PATH (str): where pretrained model to save.
    EMBEDDING_PATH (str): where pretrained embedding to save, used for evaluate embedding related metrics.
"""

import os
from os.path import dirname, realpath

from .config import Config

ROOT_PATH = dirname(dirname(dirname(realpath(__file__))))
SAVE_PATH = os.path.join(ROOT_PATH, 'save')
DATA_PATH = os.path.join(ROOT_PATH, 'data')
DATASET_PATH = os.path.join(DATA_PATH, 'dataset')
MODEL_PATH = os.path.join(DATA_PATH, 'model')
PRETRAIN_PATH = os.path.join(MODEL_PATH, 'pretrain')
EMBEDDING_PATH = os.path.join(DATA_PATH, 'embedding')
