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
            'https://pkueducn-my.sharepoint.com/:u:/g/personal/franciszhou_pkueducn_onmicrosoft_com/EdVnNcteOkpAkLdNL-ejvAABPieUd8jIty3r1jcdJvGLzw?download=1',
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
            'https://pkueducn-my.sharepoint.com/:u:/g/personal/franciszhou_pkueducn_onmicrosoft_com/EXe_sjFhfqpJoTbNcoUPJf8Bl_4U-lnduct0z8Dw5HVCPw?download=1',
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
            'https://pkueducn-my.sharepoint.com/:u:/g/personal/franciszhou_pkueducn_onmicrosoft_com/EQHOlW2m6mFEqHgt94PfoLsBbmQQeKQEOMyL1lLEHz7LvA?download=1',
            'redial_gpt2.zip',
            '15661f1cb126210a09e30228e9477cf57bbec42140d2b1029cc50489beff4eb8',
        ),
        'special_token_idx': {
            'pad': -100,
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
