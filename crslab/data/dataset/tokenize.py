# @Time   : 2022/9/28
# @Author : Xinyu Tang
# @Email  : txy20010310@163.com

import os
from nltk import word_tokenize
from transformers import AutoTokenizer
import pkuseg
import nltk
import jieba

class CrsTokenize:

    def __init__(self, path=None) -> None:
        self.path = path

        if path is not None:
            self.my_tokenizer = AutoTokenizer.from_pretrained(path)

    def tokenize(self, text, tokenizer):
        tokenize_fun = getattr(self, tokenizer + '_tokenize')
        return tokenize_fun(text)

    def nltk_tokenize(self, text):
        # nltk.download('punkt')
        return word_tokenize(text)

    def bert_tokenize(self, text):
        return self.my_tokenizer.tokenize(text)

    def gpt2_tokenize(self, text):
        return self.my_tokenizer.tokenize(text)

    def pkuseg_tokenize(self, text):
        if not hasattr(self, 'pkuseg_tokenizer'):
            self.pkuseg_tokenizer = pkuseg.pkuseg()
        return self.pkuseg_tokenizer.cut(text)

    def jieba_tokenize(self, text):        
        split_text = jieba.cut(text)
        text_list = ' '.join(split_text).split()
        return text_list