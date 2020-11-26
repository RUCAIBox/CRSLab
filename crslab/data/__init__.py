# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/24, 2020/11/26
# @Author : Kun Zhou, Xiaolei Wang
# @Email  : francis_kun_zhou@163.com, wxl1999@foxmail.com

from crslab.data.dataloader import *
from crslab.data.dataset import *

dataset_register_table = {
    'ReDial': ReDialDataset,
}

dataloader_register_table = {
    'KGSF': KGSFDataLoader,
}


def get_dataset(config, restore, save):
    """Create dataset according to :attr:`config['model']`.

    Args:
        config (Config): An instance object of Config, used to record parameter information.

    Returns:
        Dataset: Constructed dataset.
    """
    if config['dataset'] in dataset_register_table:
        return dataset_register_table[config['dataset']](config, restore, save)
    else:
        raise NotImplementedError('The dataloader [{}] has not been implemented'.format(config['dataset']))


def get_dataloader(config, dataset):
    """Return a dataloader class according to :attr:`config`.

    Args:
        name (str): The stage of dataloader. It can only take two values: 'train' or 'evaluation'.
        config (Config): An instance object of Config, used to record parameter information.
        eval_setting (EvalSetting): An instance object of EvalSetting, used to record evaluation settings.

    Returns:
        type: The dataloader class that meets the requirements in :attr:`config`.
    """
    if config['rec_model'] in dataloader_register_table:
        return dataloader_register_table[config['rec_model']](config, dataset)
    else:
        raise NotImplementedError('The dataloader [{}, {}] has not been implemented'.
                                  format(config['rec_model'], config['conv_model']))
