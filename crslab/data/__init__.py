# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/24, 2020/12/1
# @Author : Kun Zhou, Xiaolei Wang
# @Email  : francis_kun_zhou@163.com, wxl1999@foxmail.com

from crslab.data.dataloader import *
from crslab.data.dataset import *

dataset_register_table = {
    'ReDial': ReDialDataset,
}

dataloader_register_table = {
    'KGSF': KGSFDataLoader,
    'KBRD': KBRDDataLoader
}


def get_dataset(opt, restore, save) -> BaseDataset:
    """Create dataset according to :attr:`config['model']`.

    Args:
        opt (Config): An instance object of Config, used to record parameter information.

    Returns:
        Dataset: Constructed dataset.
    """
    dataset = opt['dataset']
    if dataset in dataset_register_table:
        return dataset_register_table[dataset](opt, restore, save)
    else:
        raise NotImplementedError(f'The dataloader [{dataset}] has not been implemented')


def get_dataloader(opt, dataset) -> BaseDataLoader:
    """Return a dataloader class according to :attr:`config`.

    Args:
        name (str): The stage of dataloader. It can only take two values: 'train' or 'evaluation'.
        opt (Config): An instance object of Config, used to record parameter information.
        eval_setting (EvalSetting): An instance object of EvalSetting, used to record evaluation settings.

    Returns:
        type: The dataloader class that meets the requirements.txt in :attr:`config`.
    """
    model_name = opt['model_name']
    if model_name in dataloader_register_table:
        return dataloader_register_table[model_name](opt, dataset)
    else:
        raise NotImplementedError(f'The dataloader [{model_name}] has not been implemented')
