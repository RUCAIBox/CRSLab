# -*- encoding: utf-8 -*-
# @Time    :   2020/12/21
# @Author  :   Xiaolei Wang
# @email   :   wxl1999@foxmail.com

# UPDATE
# @Time    :   2020/12/21
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
}
