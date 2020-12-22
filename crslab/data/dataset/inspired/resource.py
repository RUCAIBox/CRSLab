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
    'nltk': {
        'version': '0.22',
        'file': DownloadableFile(
            'http://d0.ananas.chaoxing.com/download/1f5ab4127c9294b7a1a0783e428b14e8?fn=nltk',
            'inspired_nltk.zip',
            '776cadc7585abdbca2738addae40488826c82de3cfd4c2dc13dcdd63aefdc5c4',
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
