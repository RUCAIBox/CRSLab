# -*- encoding: utf-8 -*-
# @Time    :   2020/12/4
# @Author  :   Xiaolei Wang
# @email   :   wxl1999@foxmail.com

# UPDATE
# @Time    :   2020/12/15
# @Author  :   Xiaolei Wang
# @email   :   wxl1999@foxmail.com

from crslab.download import DownloadableFile

resources = {
    'pkuseg': {
        'version': '0.2',
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
    }
}
