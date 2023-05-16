# -*- encoding: utf-8 -*-
# @Time    :   2021/1/6
# @Author  :   Xiaolei Wang
# @email   :   wxl1999@foxmail.com

# UPDATE
# @Time    :   2021/1/7
# @Author  :   Xiaolei Wang
# @email   :   wxl1999@foxmail.com

from crslab.download import DownloadableFile

"""Download links of pretrain models.

Now we provide the following models:

- `BERT`_: zh, en
- `GPT2`_: zh, en

.. _BERT:
   https://www.aclweb.org/anthology/N19-1423/
.. _GPT2:
   https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
    
"""

resources = {
    'bert': {
        'zh': {
            'version': '0.1',
            'file': DownloadableFile(
                'https://pkueducn-my.sharepoint.com/:u:/g/personal/franciszhou_pkueducn_onmicrosoft_com/EXm6uTgSkO1PgDD3TV9UtzMBfsAlJOun12vwB-hVkPRbXw?download=1',
                'bert_zh.zip',
                'e48ff2f3c2409bb766152dc5577cd5600838c9052622fd6172813dce31806ed3'
            )
        },
        'en': {
            'version': '0.1',
            'file': DownloadableFile(
                'https://pkueducn-my.sharepoint.com/:u:/g/personal/franciszhou_pkueducn_onmicrosoft_com/EfcnG_CkYAtKvEFUWvRF8i0BwmtCKnhnjOBwPW0W1tXqMQ?download=1',
                'bert_en.zip',
                '61b08202e8ad09088c9af78ab3f8902cd990813f6fa5b8b296d0da9d370006e3'
            )
        },
    },
    'gpt2': {
        'zh': {
            'version': '0.1',
            'file': DownloadableFile(
                'https://pkueducn-my.sharepoint.com/:u:/g/personal/franciszhou_pkueducn_onmicrosoft_com/EdwPgkE_-_BCsVSqo4Ao9D8BKj6H_0wWGGxHxt_kPmoSwA?download=1',
                'gpt2_zh.zip',
                '5f366b729e509164bfd55026e6567e22e101bfddcfaac849bae96fc263c7de43'
            )
        },
        'en': {
            'version': '0.1',
            'file': DownloadableFile(
                'https://pkueducn-my.sharepoint.com/:u:/g/personal/franciszhou_pkueducn_onmicrosoft_com/Ebe4PS0rYQ9InxmGvJ9JNXgBMI808ibQc93N-dAubtbTgQ?download=1',
                'gpt2_en.zip',
                '518c1c8a1868d4433d93688f2bf7f34b6216334395d1800d66308a80f4cac35e'
            )
        }
    }
}
