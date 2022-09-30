# @Time   : 2022/9/28
# @Author : Xinyu Tang
# @Email  : txy20010310@163.com

from transformers import AutoTokenizer

from crslab.tokenizer.base import BaseCrsTokenize

class gpt2_tokenize(BaseCrsTokenize):

    def __init__(self, path=None) -> None:
        super().__init__(path)
        self.my_tokenizer = AutoTokenizer.from_pretrained(path)

    def tokenize(self, text):
        return self.my_tokenizer.tokenize(text)