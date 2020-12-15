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
        'version': '0.1',
        'file': DownloadableFile('1zrszs2EcNlim3l7O0BH6XbalLMeUcMFv', 'kgsf_redial.zip',
                                 'f627841644a184079acde1b0185e3a223945061c3a591f4bc0d7f62e7263f548',
                                 from_google=True),
    },
    'TGReDial': {
        'version': '0.1',
        'file': DownloadableFile('1UGHZUnZ7mjPEhRVjgG2Rdgm4_p7ynPVX', 'kgsf_tgredial.zip',
                                 'da573b6789f1fb99c4b9617dd40de4fb746a933c1f577226cd0b9a72f2eaadf3',
                                 from_google=True),
    },
    'GoRecDial': {
        'version': '0.1',
        'file': DownloadableFile('1jyK2zD64UKjJniz8RWZLeBxu6gfsT4ty', 'kgsf_gorecdial.zip',
                                 '9794abf12b5d6773d867556685da14d951d42f64a5c4781af7d6fb720e87ec4f',
                                 from_google=True)
    }
}
