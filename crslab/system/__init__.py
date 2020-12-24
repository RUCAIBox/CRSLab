# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/24, 2020/12/24
# @Author : Kun Zhou, Xiaolei Wang
# @Email  : francis_kun_zhou@163.com, wxl1999@foxmail.com

from loguru import logger

from .kbrd_system import KBRDSystem
from .kgsf_system import KGSFSystem
from .redial_system import ReDialSystem
from .tgredial_system import TGReDialSystem

system_register_table = {
    'KGSF': KGSFSystem,
    'KBRD': KBRDSystem,
    'TGRec_TGConv': TGReDialSystem,
    'TGRec_TGConv_TGPolicy': TGReDialSystem,
    'ReDialRec_ReDialConv': ReDialSystem,
    'GPT2': TGReDialSystem,
    'Transformer': TGReDialSystem
}


def get_system(opt, train_dataloader, valid_dataloader, test_dataloader, vocab, side_data=None, restore=False,
               debug=False):
    """
    return the system class
    """
    model_name = opt['model_name']
    if model_name in system_register_table:
        system = system_register_table[model_name](opt, train_dataloader, valid_dataloader, test_dataloader,
                                                   vocab, side_data, restore, debug)
        logger.info(f'[Build system {model_name}]')
        return system
    else:
        raise NotImplementedError('The system with model [{}] in dataset [{}] has not been implemented'.
                                  format(model_name, opt['dataset']))
