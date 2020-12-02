# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/24, 2020/11/30
# @Author : Kun Zhou, Xiaolei Wang
# @Email  : francis_kun_zhou@163.com, wxl1999@foxmail.com

from .kgsf_system import KGSFSystem

system_register_table = {
    'KGSF': KGSFSystem
}


def get_system(opt, train_dataloader, valid_dataloader, test_dataloader, ind2token, side_data=None, debug=False):
    """
    return the system class
    """
    model_name = opt['model_name']
    if model_name in system_register_table:
        return system_register_table[model_name](opt, train_dataloader, valid_dataloader, test_dataloader,
                                                 ind2token, side_data, debug)
    else:
        raise NotImplementedError('The system with model [{}] in dataset [{}] has not been implemented'.
                                  format(model_name, opt['dataset']))
