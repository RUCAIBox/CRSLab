# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/24, 2020/12/2
# @Author : Kun Zhou, Xiaolei Wang
# @Email  : francis_kun_zhou@163.com, wxl1999@foxmail.com

from loguru import logger

from .kgsf_model import KGSFModel
from .kbrd_model import KBRDModel

Model_register_table = {
    'KGSF': KGSFModel,
    'KBRD': KBRDModel
}


def get_model(config, model_name, device, side_data=None):
    if model_name in Model_register_table:
        model = Model_register_table[model_name](config, device, side_data)
        logger.info(f'[Build model {model_name}]')
        return model
    else:
        raise NotImplementedError('Model [{}] has not been implemented'.format(model_name))
