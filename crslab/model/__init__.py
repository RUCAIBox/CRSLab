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

# only model needs side data can be written into this table
Model_SideData_table = {
    'KGSF': ["entity_kg", "word_kg"],
    'KBRD': ["entity_kg"]
}


def get_model(config, model_name, device, side_data=None):
    if model_name in Model_register_table:
        if model_name in Model_SideData_table:
            side_data.update(side_data[key] for key in Model_SideData_table[model_name])
        return Model_register_table[model_name](config, device, side_data)
    else:
        raise NotImplementedError('The model [{}] has not been implemented'.format(model_name))
