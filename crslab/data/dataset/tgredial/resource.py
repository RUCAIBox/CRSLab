# -*- encoding: utf-8 -*-
# @Time    :   2020/12/4
# @Author  :   Xiaolei Wang
# @email   :   wxl1999@foxmail.com

# UPDATE
# @Time    :   2020/12/22
# @Author  :   Xiaolei Wang
# @email   :   wxl1999@foxmail.com

from crslab.download import DownloadableFile

resources = {
    'pkuseg': {
        'version': '0.3',
        'file': DownloadableFile(
            'http://d0.ananas.chaoxing.com/download/d043702b50e547d369c81e276e6f6032?fn=pkuseg',
            'tgredial_pkuseg.zip',
            '8b7e23205778db4baa012eeb129cf8d26f4871ae98cdfe81fde6adc27a73a8d6',
        ),
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
        'version': '0.3',
        'file': DownloadableFile(
            'http://d0.ananas.chaoxing.com/download/3e5b2aa52092267ad3443f528cb1d20d?fn=bert',
            'tgredial_bert.zip',
            'd40f7072173c1dc49d4a3125f9985aaf0bd0801d7b437348ece9a894f485193b'
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
            'http://d0.ananas.chaoxing.com/download/becf68518c219e41f37191b983c62a0f?fn=gpt2',
            'tgredial_gpt2.zip',
            '2077f137b6a11c2fd523ca63b06e75cc19411cd515b7d5b997704d9e81778df9'
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
