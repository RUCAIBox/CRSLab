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
        'version': '0.3',
        'file': DownloadableFile(
            'https://pkueducn-my.sharepoint.com/:u:/g/personal/franciszhou_pku_edu_cn/ESM_Wc7sbAlOgZWo_6lOx34B6mboskdpNdB7FLuyXUET2A?download=1',
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
        'version': '0.3',
        'file': DownloadableFile(
            'https://pkueducn-my.sharepoint.com/:u:/g/personal/franciszhou_pku_edu_cn/EcTG05imCYpFiBarVfnsAfkBVsbq1iPw23CYcp9kYE9X4g?download=1',
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
        'version': '0.3',
        'file': DownloadableFile(
            'https://pkueducn-my.sharepoint.com/:u:/g/personal/franciszhou_pku_edu_cn/Edg4_nbKA49HnQPcd65gPdoBALPADQd4V5qVqOrUub2m9w?download=1',
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
