# -*- encoding: utf-8 -*-
# @Time    :   2020/12/21
# @Author  :   Xiaolei Wang
# @email   :   wxl1999@foxmail.com

# UPDATE
# @Time    :   2020/12/22
# @Author  :   Xiaolei Wang
# @email   :   wxl1999@foxmail.com

from crslab.download import DownloadableFile

resources = {
    'nltk': {
        'version': '0.3',
        'file': DownloadableFile(
            'https://pkueducn-my.sharepoint.com/:u:/g/personal/franciszhou_pkueducn_onmicrosoft_com/ESB7grlJlehKv7XmYgMgq5AB85LhRu_rSW93_kL8Arfrhw?download=1',
            'opendialkg_nltk.zip',
            '6487f251ac74911e35bec690469fba52a7df14908575229b63ee30f63885c32f'
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
    'bert': {
        'version': '0.3',
        'file': DownloadableFile(
            'https://pkueducn-my.sharepoint.com/:u:/g/personal/franciszhou_pkueducn_onmicrosoft_com/EWab0Pzgb4JOiecUHZxVaEEBRDBMoeLZDlStrr7YxentRA?download=1',
            'opendialkg_bert.zip',
            '0ec3ff45214fac9af570744e9b5893f224aab931744c70b7eeba7e1df13a4f07'
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
        'version': '0.3',
        'file': DownloadableFile(
            'https://pkueducn-my.sharepoint.com/:u:/g/personal/franciszhou_pkueducn_onmicrosoft_com/EdE5iyKIoAhLvCwwBN4MdJwB2wsDADxJCs_KRaH-G3b7kg?download=1',
            'opendialkg_gpt2.zip',
            'dec20b01247cfae733988d7f7bfd1c99f4bb8ba7786b3fdaede5c9a618c6d71e'
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
