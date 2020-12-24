# -*- encoding: utf-8 -*-
# @Time    :   2020/12/13
# @Author  :   Xiaolei Wang
# @email   :   wxl1999@foxmail.com

# UPDATE
# @Time    :   2020/12/24
# @Author  :   Xiaolei Wang
# @email   :   wxl1999@foxmail.com

from crslab.download import DownloadableFile

resources = {
    'zh': {
        'version': '0.25',
        'file': DownloadableFile(
            'http://d0.ananas.chaoxing.com/download/40feec1c3fc247ec7da6096da22e9a85?fn=tgredial',
            'bert-gpt2_zh.zip',
            'f8ca711a150c0483e1016d642984863158b81b1dfb2d9d07239209024808e75e'
        )
    },
    'en': {
        'version': '0.25',
        'file': DownloadableFile(
            'http://d0.ananas.chaoxing.com/download/6c91d43b81bc0b5e2f81e0b2be57128a?fn=redial',
            'bert-gpt2_en.zip',
            'f71c134ce9cf9c0c8b4af9733bd6ce5bdf7ccd6e979ab0a68b693f19224554ad'),
    },
}
