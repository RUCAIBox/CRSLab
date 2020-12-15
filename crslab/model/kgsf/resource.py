# -*- encoding: utf-8 -*-
# @Time    :   2020/12/13
# @Author  :   Xiaolei Wang
# @email   :   wxl1999@foxmail.com

# UPDATE
# @Time    :   2020/12/15
# @Author  :   Xiaolei Wang
# @email   :   wxl1999@foxmail.com

from crslab.download import DownloadableFile

resources = {
    'ReDial': {
        'version': '0.2',
        'file': DownloadableFile(
            'http://d0.ananas.chaoxing.com/download/29bf59eed27f74ba6b6d16594f1bcc02?fn=redial',
            'kgsf_redial.zip',
            'f627841644a184079acde1b0185e3a223945061c3a591f4bc0d7f62e7263f548',
        ),
    },
    'TGReDial': {
        'version': '0.2',
        'file': DownloadableFile(
            'http://d0.ananas.chaoxing.com/download/959f40436a07df23dd951a0c0f0d3d62?fn=tgredial',
            'kgsf_tgredial.zip',
            'c9d054b653808795035f77cb783227e6e9a938e5bedca4d7f88c6dfb539be5d1',
        ),
    },
    'GoRecDial': {
        'version': '0.1',
        'file': DownloadableFile(
            'http://d0.ananas.chaoxing.com/download/3e7de02a53df515e1444417da4d82e42?fn=gorecdial',
            'kgsf_gorecdial.zip',
            '9794abf12b5d6773d867556685da14d951d42f64a5c4781af7d6fb720e87ec4f',
        )
    }
}
