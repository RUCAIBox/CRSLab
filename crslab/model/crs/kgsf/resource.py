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
    },
    'OpenDialKG': {
        'version': '0.1',
        'file': DownloadableFile(
            'http://d0.ananas.chaoxing.com/download/f364604c91b9ed61771b1b3ee09b4cc4?fn=opendialkg',
            'kgsf_opendialkg.zip',
            '89b785b23478b1d91d6ab4f34a3658e82b52dcbb73828713a9b369fa49db9e61'
        )
    },
    'Inspired': {
        'version': '0.1',
        'file': DownloadableFile(
            'http://d0.ananas.chaoxing.com/download/7b2418b9f681bea8d6ed01bdf8ec3630?fn=inspired',
            'kgsf_inspired.zip',
            '23dfc031a3c71f2a52e29fe0183e1a501771b8d431852102ba6fd83d971f928d'
        )
    },
    'DuRecDial': {
        'version': '0.1',
        'file': DownloadableFile(
            'http://d0.ananas.chaoxing.com/download/6127c64e70d80d8d7fd2cac7bc4b7067?fn=durecdial',
            'kgsf_durecdial.zip',
            'f9a39c2382efe88d80ef14d7db8b4cbaf3a6eb92a33e018dfc9afba546ba08ef'
        )
    }
}
