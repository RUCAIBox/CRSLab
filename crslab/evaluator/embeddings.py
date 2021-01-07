# -*- encoding: utf-8 -*-
# @Time    :   2020/12/18
# @Author  :   Xiaolei Wang
# @email   :   wxl1999@foxmail.com

# UPDATE
# @Time    :   2020/12/18
# @Author  :   Xiaolei Wang
# @email   :   wxl1999@foxmail.com

from crslab.download import DownloadableFile

resources = {
    'zh': {
        'version': '0.2',
        'file': DownloadableFile(
            'http://d0.ananas.chaoxing.com/download/6a46e2a711effa3872f82cec4d92d92c?fn=cc.zh.300',
            'cc.zh.300.zip',
            'effd9806809a1db106b5166b817aaafaaf3f005846f730d4c49f88c7a28a0ac3'
        )
    },
    'en': {
        'version': '0.2',
        'file': DownloadableFile(
            'http://d0.ananas.chaoxing.com/download/701505c10567b5f715a35c405b1fe8e5?fn=cc.en.300',
            'cc.en.300.zip',
            '96a06a77da70325997eaa52bfd9acb1359a7c3754cb1c1aed2fc27c04936d53e'
        )
    }
}
