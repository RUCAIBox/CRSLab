# @Time   : 2022/9/30
# @Author : Xinyu Tang
# @Email  : txy20010310@163.com

import os
from transformers import AutoTokenizer

class BaseCrsTokenize:

    def __init__(self, path=None) -> None:
        pass

    def tokenize(self, text):
        '''
        split token
        '''
        pass