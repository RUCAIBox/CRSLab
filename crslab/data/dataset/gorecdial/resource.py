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
        'version': '0.21',
        'file': DownloadableFile(
            'http://d0.ananas.chaoxing.com/download/fc9d96c44806d458f4f42ed336f0d239?fn=nltk',
            'gorecdial_nltk.zip',
            '7e523f7ca90bb32ee8f2471ac5736717c45b20822c63bd958d0546de0a9cd863',
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
        'version': '0.25',
        'file': DownloadableFile(
            'http://d0.ananas.chaoxing.com/download/d9c00b7daee6c9f954130ddbc1797cac?fn=bert',
            'gorecdial_bert.zip',
            'fc7aff18504f750d8974d90f2941a01ff22cc054283124936b778ba91f03554f'
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
        'version': '0.25',
        'file': DownloadableFile(
            'http://d0.ananas.chaoxing.com/download/c4cb3cabbf78c5edb2f6dbb36fdadcbd?fn=gpt2',
            'gorecdial_gpt2.zip',
            '7234138dcc27ed00bdac95da4096cd435023c229d227fa494d2bd7a653a492a9'
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
