# @Time   : 2021/8/2
# @Author : Chenzhan Shang
# @Email  : czshang@outlook.com

r"""
Last.FM
======
References:
    Last.FM Dataset:
    https://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip

"""

import os

from crslab.config import DATASET_PATH
from crslab.dataset import AttributeBaseDataset


class LastFMDataset(AttributeBaseDataset):
    """

    Attributes:
        train_data: train dataset.
        valid_data: valid dataset.
        test_data: test dataset.

    """
    def __init__(self, opt, tokenize, restore=False, save=False):
        resource = None
        dpath = os.path.join(DATASET_PATH, "lastfm")
        super().__init__(opt, dpath, resource, restore, save)


