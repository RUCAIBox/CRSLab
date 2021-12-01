# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/24, 2020/12/29
# @Author : Kun Zhou, Xiaolei Wang
# @Email  : francis_kun_zhou@163.com, wxl1999@foxmail.com

# @Time    :   2021/10/6
# @Author  :   Zhipeng Zhao
# @email   :   oran_official@outlook.com



from loguru import logger

from .inspired import InspiredSystem
from .kbrd import KBRDSystem
from .kgsf import KGSFSystem
from .redial import ReDialSystem
from .ntrd import NTRDSystem
from .tgredial import TGReDialSystem

system_register_table = {
    'ReDialRec_ReDialConv': ReDialSystem,
    'KBRD': KBRDSystem,
    'KGSF': KGSFSystem,
    'TGRec_TGConv': TGReDialSystem,
    'TGRec_TGConv_TGPolicy': TGReDialSystem,
    'InspiredRec_InspiredConv': InspiredSystem,
    'GPT2': TGReDialSystem,
    'Transformer': TGReDialSystem,
    'ConvBERT': TGReDialSystem,
    'ProfileBERT': TGReDialSystem,
    'TopicBERT': TGReDialSystem,
    'PMI': TGReDialSystem,
    'MGCG': TGReDialSystem,
    'BERT': TGReDialSystem,
    'SASREC': TGReDialSystem,
    'GRU4REC': TGReDialSystem,
    'Popularity': TGReDialSystem,
    'TextCNN': TGReDialSystem,
    'NTRD': NTRDSystem
}


def get_system(opt, train_dataloader, valid_dataloader, test_dataloader, vocab, side_data, restore_system=False,
               interact=False, debug=False, tensorboard=False):
    """
    return the system class
    """
    model_name = opt['model_name']
    if model_name in system_register_table:
        system = system_register_table[model_name](opt, train_dataloader, valid_dataloader, test_dataloader, vocab,
                                                   side_data, restore_system, interact, debug, tensorboard)
        logger.info(f'[Build system {model_name}]')
        return system
    else:
        raise NotImplementedError('The system with model [{}] in dataset [{}] has not been implemented'.
                                  format(model_name, opt['dataset']))
