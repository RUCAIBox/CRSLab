# @Time   : 2022/9/28
# @Author : Xinyu Tang
# @Email  : txy20010310@163.com

from transformers import AutoTokenizer

from crslab.data.dataset.tokenizer.base import BaseTokenizer


class Gpt2Tokenizer(BaseTokenizer):

    def __init__(self, path=None) -> None:
        self.special_token_idx = {
            'pad': 0,
            'start': 1,
            'end': 2,
            'unk': 3,
            'cls': 101,
            'sep': 102,
            'sent_split': 4,
            'word_split': 5,
            'pad_entity': 0,
            'pad_word': 0,
            'pad_topic': 0,
        }
        self.my_tokenizer = AutoTokenizer.from_pretrained(path)
        super().__init__(path)        

    def tokenize(self, text):
        return self.my_tokenizer.tokenize(text)
