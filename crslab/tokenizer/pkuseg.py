# @Time   : 2022/9/30
# @Author : Xinyu Tang
# @Email  : txy20010310@163.com

import pkuseg

from crslab.tokenizer.base import BaseCrsTokenize

class pkuseg_tokenize(BaseCrsTokenize):

    def __init__(self, path=None) -> None:
        self.pkuseg_tokenizer = pkuseg.pkuseg()
        super().__init__(path)

    def tokenize(self, text):        
        return self.pkuseg_tokenizer.cut(text)