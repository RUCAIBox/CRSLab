# @Time   : 2022/9/30
# @Author : Xinyu Tang
# @Email  : txy20010310@163.com

from transformers import AutoTokenizer

from crslab.data.dataset.tokenizer.base import BaseTokenizer


class bert_tokenize(BaseTokenizer):

    def __init__(self, path=None) -> None:
        self.special_token_idx =  {
            'pad': 0,
            'start': 101,
            'end': 102,
            'unk': 100,
            'sent_split': 2,
            'word_split': 3,
            'pad_entity': 0,
            'pad_word': 0,
            'pad_topic': 0
        }
        self.my_tokenizer = AutoTokenizer.from_pretrained(path)
        super().__init__(path)

    def tokenize(self, text):
        return self.my_tokenizer.tokenize(text)
