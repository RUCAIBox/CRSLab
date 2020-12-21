# -*- encoding: utf-8 -*-
# @Time    :   2020/12/4
# @Author  :   Xiaolei Wang
# @email   :   wxl1999@foxmail.com

# UPDATE
# @Time    :   2020/12/21
# @Author  :   Xiaolei Wang
# @email   :   wxl1999@foxmail.com

from crslab.download import DownloadableFile

resources = {
    'pkuseg': {
        'version': '0.21',
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
        'version': '0.21',
        'file': DownloadableFile(
            'http://d0.ananas.chaoxing.com/download/d5575fb95159105a87c06d95ff319b8b?fn=bert',
            'tgredial_bert.zip',
            'e50016a22d250492666de013b79191d71eabef130536e9fa0315bf80d180897b'
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
        'version': '0.21',
        'file': DownloadableFile(
            'http://d0.ananas.chaoxing.com/download/8a1f9568aa47e5b4f65044f9d0490b43?fn=gpt2',
            'tgredial_gpt2.zip',
            '5218a580debd65cefbe4097edb7c03d6bf139a08ec0f3897d1c09333a1e5d226'
        ),
        'special_token_idx': {
            'pad': 0,
            'start': 1,
            'end': 2,
            'unk': 3,
            'sent_split': 4,
            'word_split': 5,
            'pad_entity': 0,
            'pad_word': 0,
            'pad_topic': 0
        },
    }
}
