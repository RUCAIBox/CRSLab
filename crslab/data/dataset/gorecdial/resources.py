# -*- encoding: utf-8 -*-
# @Time    :   2020/12/14
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
            'https://pkueducn-my.sharepoint.com/:u:/g/personal/franciszhou_pku_edu_cn/EYmobnFBox1LnGKGW4TMCk8BW6rnjdAZNVsNo8uJ8ZsJLg?download=1',
            'gorecdial.zip',
            '66035bf24862535a072cc6778a3affd541ae0a4aa1fe31455d4fb063b301f087',
        ),
        'nltk': {
            'special_token_idx': {
                'pad': 0,
                'start': 1,
                'end': 2,
                'unk': 3,
                'pad_entity': 0,
                'pad_word': 0,
                'pad_topic': 0
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
            }
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
        },
    },

}
