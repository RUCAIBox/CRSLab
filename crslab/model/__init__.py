# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/24
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

from .kgsf_model import KGSFModel

Model_register_table = {
    'KGSF': KGSFModel
}


def get_model(config, model_name, device, side_data=None):
    if model_name in Model_register_table:
        return Model_register_table[model_name](config, device, side_data)
    else:
        raise NotImplementedError('Model [{}] has not been implemented'.format(model_name))
