# -*- encoding: utf-8 -*-
# @Time    :   2020/12/21
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
            'http://d0.ananas.chaoxing.com/download/3cc0807922aa85b7ec2a331b13845353?fn=nltk',
            'opendialkg_nltk.zip',
            '6487f251ac74911e35bec690469fba52a7df14908575229b63ee30f63885c32f',
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
        'version': '0.25',
        'file': DownloadableFile(
            'http://d0.ananas.chaoxing.com/download/e50849698023f3b006129c1fc9c02e22?fn=bert',
            'opendialkg_bert.zip',
            '0ec3ff45214fac9af570744e9b5893f224aab931744c70b7eeba7e1df13a4f07'
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
            'http://d0.ananas.chaoxing.com/download/e3b05fa67a7900f2cd44602aef359810?fn=gpt2',
            'opendialkg_gpt2.zip',
            'dec20b01247cfae733988d7f7bfd1c99f4bb8ba7786b3fdaede5c9a618c6d71e'
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
