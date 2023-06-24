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
            'https://pkueducn-my.sharepoint.com/:u:/g/personal/franciszhou_pkueducn_onmicrosoft_com/Ee7FleGfEStCimV4XRKvo-kBR8ABdPKo0g_XqgLJPxP6tg?download=1',
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
            'https://pkueducn-my.sharepoint.com/:u:/g/personal/franciszhou_pkueducn_onmicrosoft_com/ETC9vIeFtOdElXL10Hbh4L0BGm20-lckCJ3a4u7VFCzpIg?download=1',
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
            'https://pkueducn-my.sharepoint.com/:u:/g/personal/franciszhou_pkueducn_onmicrosoft_com/EcVEcxrDMF1BrbOUD8jEXt4BJeCzUjbNFL6m6UY5W3Hm3g?download=1',
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
