# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/24, 2020/12/24, 2021/8/4
# @Author : Kun Zhou, Xiaolei Wang, Chenzhan Shang
# @Email  : francis_kun_zhou@163.com, wxl1999@foxmail.com, czshang@outlook.com

import torch
from loguru import logger

from crslab.register import model_register_table


def get_model(config, model_name, device, vocab, side_data=None):
    if model_name in model_register_table:
        model = model_register_table[model_name](config, device, vocab, side_data)
        logger.info(f'[Build model {model_name}]')
        if config.opt["gpu"] == [-1]:
            return model
        else:
            if len(config.opt["gpu"]) > 1:
                if model_name == 'PMI' or model_name == 'KBRD':
                    logger.info(f'[PMI/KBRD model does not support multi GPUs yet, using single GPU now]')
                    return model.to(device)
            return torch.nn.DataParallel(model, device_ids=config["gpu"])

    else:
        raise NotImplementedError('Model [{}] has not been implemented'.format(model_name))
