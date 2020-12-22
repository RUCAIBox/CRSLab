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
        'version': '0.22',
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
}
