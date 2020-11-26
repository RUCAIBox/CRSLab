# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/24
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

from crslab.model.base_model import *
from crslab.model.kgsf_model import *

def get_model(config, model_name, device, side_data=None):
    Model_register_table = {
        'KGSF': KGSFModel
    }
    #only model needs side data can be written into this table
    Model_SideData_table={
        'KGSF': [0,1],
        'KBRD': [0]
    }
    if model_name in Model_register_table:
        if model_name in Model_SideData_table:
            side_data=[side_data[idx] for idx in Model_SideData_table[model_name]]
            return Model_register_table[model_name](config, device, side_data)
        else:
            return Model_register_table[model_name](config, device)
    else:
        raise NotImplementedError('The model [{}] has not been implemented'.format(model_name))
