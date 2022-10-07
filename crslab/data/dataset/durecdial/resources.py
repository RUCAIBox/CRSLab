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
            'https://pkueducn-my.sharepoint.com/:u:/g/personal/franciszhou_pku_edu_cn/ERN4GhkC-fBLk1gRKZeHgo4BnQglDxv7VTVmbqgPdL108A?download=1',
            'durecdial.zip',
            '9b781f82a9192e96a1e7a9f7501edc930e0e13c0732faf8e3964360a6d5c6ca5',
        ),
        'jieba': {
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
                'pad_topic': 0
            },
        },
        'gpt2': {
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
    },
}
