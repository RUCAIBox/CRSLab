# @Time   : 2022/9/30
# @Author : Xinyu Tang
# @Email  : txy20010310@163.com

import jieba

from crslab.data.dataset.tokenizer.base import BaseTokenizer


class JiebaTokenizer(BaseTokenizer):

    def __init__(self, path=None) -> None:
        self.special_token_idx = {
            'pad': 0,
            'start': 1,
            'end': 2,
            'unk': 3,
            'pad_entity': 0,
            'pad_word': 0,
        }
        super().__init__(path)

    def tokenize(self, text):
        split_text = jieba.cut(text)
        text_list = ' '.join(split_text).split()
        return text_list
