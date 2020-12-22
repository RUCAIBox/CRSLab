# -*- encoding: utf-8 -*-
# @Time    :   2020/12/22
# @Author  :   Xiaolei Wang
# @email   :   wxl1999@foxmail.com

# UPDATE
# @Time    :   2020/12/22
# @Author  :   Xiaolei Wang
# @email   :   wxl1999@foxmail.com

from crslab.download import DownloadableFile

resources = {
    'nltk': {
        'version': '0.22',
        'file': DownloadableFile(
            'http://d0.ananas.chaoxing.com/download/1f5ab4127c9294b7a1a0783e428b14e8?fn=nltk',
            'inspired_nltk.zip',
            '776cadc7585abdbca2738addae40488826c82de3cfd4c2dc13dcdd63aefdc5c4',
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
        'version': '0.25',
        'file': DownloadableFile(
            'http://d0.ananas.chaoxing.com/download/868102c5b21c4fa7f8b3f4574b3acf97?fn=bert',
            'inspired_bert.zip',
            '9affea30978a6cd48b8038dddaa36f4cb4d8491cf8ae2de44a6d3dde2651f29c'
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
        'version': '0.25',
        'file': DownloadableFile(
            'http://d0.ananas.chaoxing.com/download/854ef833f1365ac9c7f1254f767df81d?fn=gpt2',
            'inspired_gpt2.zip',
            '23bb4ce3299186630fdf673e17f43ee43e91573ea786c922e3527e4c341a313c'
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
