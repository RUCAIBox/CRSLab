# -*- encoding: utf-8 -*-
# @Time    :   2020/12/14
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
            'https://pkueducn-my.sharepoint.com/:u:/g/personal/franciszhou_pkueducn_onmicrosoft_com/ESIqjwAg0ItAu7WGfukIt3cBXjzi7AZ9L_lcbFT1aS1qYQ?download=1',
            'gorecdial_nltk.zip',
            '58cd368f8f83c0c8555becc314a0017990545f71aefb7e93a52581c97d1b8e9b',
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
        'version': '0.31',
        'file': DownloadableFile(
            'https://pkueducn-my.sharepoint.com/:u:/g/personal/franciszhou_pkueducn_onmicrosoft_com/Ed1HT8gzvRpDosVT83BEj5QBnzKpjR3Zbf5u49yyWP-k6Q?download=1',
            'gorecdial_bert.zip',
            '4fa10c3fe8ba538af0f393c99892739fcb376d832616aa7028334c594b3fec10'
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
        }
    },
    'gpt2': {
        'version': '0.31',
        'file': DownloadableFile(
            'https://pkueducn-my.sharepoint.com/:u:/g/personal/franciszhou_pkueducn_onmicrosoft_com/EUJOHmX8v79DkZMq0x5r9d4B0UJlfw85v-VdciwKfAhpng?download=1',
            'gorecdial_gpt2.zip',
            '44a15637e014b2e6628102ff654e1aef7ec1cbfa34b7ada1a03f294f72ddd4b1'
        ),
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
