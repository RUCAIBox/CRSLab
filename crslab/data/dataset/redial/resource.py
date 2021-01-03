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
        'version': '0.31',
        'file': DownloadableFile(
            'http://d0.ananas.chaoxing.com/download/417f6ac16282e4910fc93973e954ab42?fn=nltk',
            'redial_nltk.zip',
            '01dc2ebf15a0988a92112daa7015ada3e95d855e80cc1474037a86e536de3424',
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
        'version': '0.31',
        'file': DownloadableFile(
            'http://d0.ananas.chaoxing.com/download/34869653883b3eeb5d8d7c261207680e?fn=bert',
            'redial_bert.zip',
            'fb55516c22acfd3ba073e05101415568ed3398c86ff56792f82426b9258c92fd',
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
        'version': '0.31',
        'file': DownloadableFile(
            'http://d0.ananas.chaoxing.com/download/bd10bcbc0f0bfe74dcc48fafaf518ae5?fn=gpt2',
            'redial_gpt2.zip',
            '37b1a64032241903a37b5e014ee36e50d09f7e4a849058688e9af52027a3ac36',
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
