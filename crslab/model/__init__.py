# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/24, 2020/12/16
# @Author : Kun Zhou, Xiaolei Wang
# @Email  : francis_kun_zhou@163.com, wxl1999@foxmail.com

from loguru import logger

from .kbrd import *
from .kgsf import *
from .redial import *
from .tgredial import *

Model_register_table = {
    'KGSF': KGSFModel,
    'KBRD': KBRDModel,
    'TGRec': TGRecModel,
    'TGConv': TGConvModel,
    'TGPolicy': TGPolicyModel,
    'ReDialRec': ReDialRecModel,
    'ReDialConv': ReDialConvModel
}


def get_model(config, model_name, device, vocab, side_data=None):
    if model_name in Model_register_table:
        model = Model_register_table[model_name](config, device, vocab, side_data)
        logger.info(f'[Build model {model_name}]')
        return model
    else:
        raise NotImplementedError('Model [{}] has not been implemented'.format(model_name))
