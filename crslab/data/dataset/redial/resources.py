# -*- encoding: utf-8 -*-
# @Time    :   2020/12/1
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
            'https://pkueducn-my.sharepoint.com/:u:/g/personal/franciszhou_pku_edu_cn/Ea4PEMnyyqxAl6tiAC17BcgBW8fZ6eveNKAbAU5sYt8-PQ?download=1',
            'redial.zip',
            '9fcccc47095c6c8764a3f92e9ec993a2f5f635458836ac3314dcf007ad80d639',
        ),
        'nltk':{
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
    },
}
