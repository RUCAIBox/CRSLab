# -*- encoding: utf-8 -*-
# @Time    :   2020/12/1
# @Author  :   Xiaolei Wang
# @email   :   wxl1999@foxmail.com

# UPDATE
# @Time    :   2020/12/22
# @Author  :   Xiaolei Wang
# @email   :   wxl1999@foxmail.com

from crslab.download import DownloadableFile

resources = {
    'nltk': {
        'version': '0.21',
        'file': DownloadableFile(
            'http://d0.ananas.chaoxing.com/download/2e547e232fd2795e97d9df9da6b20e30?fn=nltk',
            'redial_nltk.zip',
            '9a0317a102675b748cb31fe0e77e152c2b2e6ece3268bcade2a0cd549e99bba1',
        ),
        'special_token_idx': {
            'pad': 0,
            'start': 1,
            'end': 2,
            'unk': 3,
            'pad_entity': 0,
            'pad_word': 0
        },
    },
    'bert': {
        'version': '0.25',
        'file': DownloadableFile(
            'http://d0.ananas.chaoxing.com/download/b96b62df9ff2dceac18d76d21e7ae9e9?fn=bert',
            'redial_bert.zip',
            '21992cc07524ac5428d875e564582b84a78d262eae589a3fa164f479a1b96993',
        ),
        'special_token_idx': {
            'pad': 0,
            'start': 101,
            'end': 102,
            'unk': 100,
            'sent_split': 2,
            'word_split': 3,
            'pad_entity': 0,
            'pad_word': 0,
        },
    },
    'gpt2': {
        'version': '0.25',
        'file': DownloadableFile(
            'http://d0.ananas.chaoxing.com/download/959dd4bb927d0d47283d4d9de8179a9f?fn=gpt2',
            'redial_gpt2.zip',
            'bb0deb54766fc8a48e82697141e502dbda8e0011c198d5f051ff9ae7f0d278c3',
        ),
        'special_token_idx': {
            'pad': 0,
            'start': 1,
            'end': 2,
            'unk': 3,
            'sent_split': 4,
            'word_split': 5,
            'pad_entity': 0,
            'pad_word': 0
        },
    }
}
