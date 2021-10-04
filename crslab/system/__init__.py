# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/24, 2020/12/29, 2021/8/4
# @Author : Kun Zhou, Xiaolei Wang, Chenzhan Shang
# @Email  : francis_kun_zhou@163.com, wxl1999@foxmail.com, czshang@outlook.com

from loguru import logger

from crslab.system.inspired import InspiredSystem
from crslab.system.kbrd import KBRDSystem
from crslab.system.kgsf import KGSFSystem
from crslab.system.redial import ReDialSystem
from crslab.system.tgredial import TGReDialSystem

system_register_table = {
    'ReDial': ReDialSystem,
    'KBRD': KBRDSystem,
    'KGSF': KGSFSystem,
    'TGRec_TGConv': TGReDialSystem,
    'TGRec_TGConv_TGPolicy': TGReDialSystem,
    'InspiredRec_InspiredConv': InspiredSystem
}


def get_system(opt, agent, restore_model=False, save_model=False, interaction=False, tensorboard=False):
    """
    return the system class
    """
    model_name = opt['model_name']
    if model_name in system_register_table:
        system = system_register_table[model_name](opt, agent, restore_model, save_model, interaction, tensorboard)
        logger.info(f'[Build system {model_name}]')
        return system
    else:
        raise NotImplementedError('The system with model [{}] in dataset [{}] has not been implemented'.
                                  format(model_name, opt['dataset']))
