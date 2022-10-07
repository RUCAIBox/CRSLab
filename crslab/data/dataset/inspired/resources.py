# -*- encoding: utf-8 -*-
# @Time    :   2020/12/22
# @Author  :   Xiaolei Wang
# @email   :   wxl1999@foxmail.com

# UPDATE
# @Time    :   2020/12/22
# @Author  :   Xiaolei Wang
# @email   :   wxl1999@foxmail.com

# UPDATE
# @Time    :   2022/9/26
# @Author  :   Xinyu Tang
# @email   :   txy20010310@163.com

from crslab.download import DownloadableFile

resources = {
    'resource': {
        'version': '1.0',
        'file': DownloadableFile(
            'https://pkueducn-my.sharepoint.com/:u:/g/personal/franciszhou_pku_edu_cn/EXv8zwgCOY1EstHNjjs194cBqMIrdg4yxcyNsHKltTzyig?download=1',
            'inspired.zip',
            '1085c2ab31fd7691f24531f9beef9016b0f3137366495784569a63f82ddd95ed',
        ),
        'nltk': {
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
}
