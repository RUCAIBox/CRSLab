# @Time   : 2022/9/30
# @Author : Xinyu Tang
# @Email  : txy20010310@163.com

import nltk
from nltk import word_tokenize

from crslab.data.dataset.tokenizer.base import BaseTokenizer


class nltk_tokenize(BaseTokenizer):

    def __init__(self, path=None) -> None:
        self.special_token_idx = {
            'pad': 0,
            'start': 1,
            'end': 2,
            'unk': 3,
            'pad_entity': 0,
            'pad_word': 0,
            'pad_topic': 0,
        }
        super().__init__(path)

    def tokenize(self, text):
        nltk.download('punkt')
        return word_tokenize(text)
