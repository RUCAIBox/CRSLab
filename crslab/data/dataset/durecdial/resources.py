# -*- encoding: utf-8 -*-
# @Time    :   2020/12/22
# @Author  :   Xiaolei Wang
# @email   :   wxl1999@foxmail.com

# UPDATE
# @Time    :   2020/12/22
# @Author  :   Xiaolei Wang
# @email   :   wxl1999@foxmail.com

from crslab.download import DownloadableFile

resources = {
    'jieba': {
        'version': '0.3',
        'file': DownloadableFile(
            'http://d0.ananas.chaoxing.com/download/4079f3663051e87d64c4223e522394e6?fn=jieba',
            'durecdial_jieba.zip',
            'c2d24f7d262e24e45a9105161b5eb15057c96c291edb3a2a7b23c9c637fd3813',
        ),
        'special_token_idx': {
            'pad': 0,
            'start': 1,
            'end': 2,
            'unk': 3,
            'pad_entity': 0,
            'pad_word': 0,
        },
    },
    'bert': {
        'version': '0.3',
        'file': DownloadableFile(
            'http://d0.ananas.chaoxing.com/download/21775beb2a83dc33395d1d7b1311b0bb?fn=bert',
            'durecdial_bert.zip',
            '0126803aee62a5a4d624d8401814c67bee724ad0af5226d421318ac4eec496f5'
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
            'pad_topic': 0
        },
    },
    'gpt2': {
        'version': '0.3',
        'file': DownloadableFile(
            'http://d0.ananas.chaoxing.com/download/742d3517069217c2b901f68a9055953d?fn=gpt2',
            'durecdial_gpt2.zip',
            'a7a93292b4e4b8a5e5a2c644f85740e625e04fbd3da76c655150c00f97d405e4'
        ),
        'special_token_idx': {
            'pad': 0,
            'start': 101,
            'end': 102,
            'unk': 100,
            'cls': 101,
            'sep': 102,
            'sent_split': 2,
            'word_split': 3,
            'pad_entity': 0,
            'pad_word': 0,
            'pad_topic': 0,
        },
    }
}
