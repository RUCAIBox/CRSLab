# -*- encoding: utf-8 -*-
# @Time    :   2020/12/22
# @Author  :   Xiaolei Wang
# @email   :   wxl1999@foxmail.com

# UPDATE
# @Time    :   2020/12/29
# @Author  :   Xiaolei Wang
# @email   :   wxl1999@foxmail.com

import os
from os.path import dirname, realpath

from .config import Config

ROOT_PATH = dirname(dirname(dirname(realpath(__file__))))
SAVE_PATH = os.path.join(ROOT_PATH, 'save')
DATA_PATH = os.path.join(ROOT_PATH, 'data')
DATASET_PATH = os.path.join(DATA_PATH, 'dataset')
MODEL_PATH = os.path.join(DATA_PATH, 'model')
EMBEDDING_PATH = os.path.join(DATA_PATH, 'embedding')
