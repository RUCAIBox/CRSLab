# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/24, 2020/12/29, 2021/8/4
# @Author : Kun Zhou, Xiaolei Wang, Chenzhan Shang
# @Email  : francis_kun_zhou@163.com, wxl1999@foxmail.com, czshang@outlook.com

from loguru import logger

from crslab.register import system_register_table


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
