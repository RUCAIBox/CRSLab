# @Time   : 2022/9/30
# @Author : Xinyu Tang
# @Email  : txy20010310@163.com

from crslab.data.dataset.tokenizer.base import BaseCrsTokenize

import jieba


class jieba_tokenize(BaseCrsTokenize):

    def __init__(self, path=None) -> None:
        super().__init__(path)

    def tokenize(self, text):
        split_text = jieba.cut(text)
        text_list = ' '.join(split_text).split()
        return text_list
