# -*- encoding: utf-8 -*-
# @Time    :   2020/12/14
# @Author  :   Xiaolei Wang
# @email   :   wxl1999@foxmail.com

# UPDATE
# @Time    :   2020/12/15
# @Author  :   Xiaolei Wang
# @email   :   wxl1999@foxmail.com

from crslab.download import DownloadableFile

resources = {
    'nltk': {
        'version': '0.11',
        'file': DownloadableFile('1LONCv9G50A933_muSYJrVFhj9TsPeOBe', 'gorecdial_nltk.zip',
                                 '7e523f7ca90bb32ee8f2471ac5736717c45b20822c63bd958d0546de0a9cd863',
                                 from_google=True),
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
