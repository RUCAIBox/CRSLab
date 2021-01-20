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
                'http://d0.ananas.chaoxing.com/download/c7b8011751cf08a3f83c1f63593c657f?fn=zh',
                'bert_zh.zip',
                'e48ff2f3c2409bb766152dc5577cd5600838c9052622fd6172813dce31806ed3'
            )
        },
        'en': {
            'version': '0.1',
            'file': DownloadableFile(
                'http://d0.ananas.chaoxing.com/download/1881a7059d8e5e5c4150fb80c21856cd?fn=en',
                'bert_en.zip',
                '61b08202e8ad09088c9af78ab3f8902cd990813f6fa5b8b296d0da9d370006e3'
            )
        },
    },
    'gpt2': {
        'zh': {
            'version': '0.1',
            'file': DownloadableFile(
                'http://d0.ananas.chaoxing.com/download/ce9ef13619fe0fb65a5434a3f83986b0?fn=zh',
                'gpt2_zh.zip',
                '5f366b729e509164bfd55026e6567e22e101bfddcfaac849bae96fc263c7de43'
            )
        },
        'en': {
            'version': '0.1',
            'file': DownloadableFile(
                'http://d0.ananas.chaoxing.com/download/8ea87b8bb9f6edf8154ea490b5b7ae01?fn=en',
                'gpt2_en.zip',
                '518c1c8a1868d4433d93688f2bf7f34b6216334395d1800d66308a80f4cac35e'
            )
        }
    }
}
