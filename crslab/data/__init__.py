# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/24, 2020/12/29, 2020/12/17
# @Author : Kun Zhou, Xiaolei Wang, Yuanhang Zhou
# @Email  : francis_kun_zhou@163.com, wxl1999@foxmail.com, sdzyh002@gmail.com

# @Time   : 2021/10/06
# @Author : Zhipeng Zhao
# @Email  : oran_official@outlook.com

"""Data module which reads, processes and batches data for the whole system

Attributes:
    dataset_register_table (dict): record all supported dataset
    dataset_language_map (dict): record all dataset corresponding language
    dataloader_register_table (dict): record all model corresponding dataloader

"""

from crslab.data.dataloader import *
from crslab.data.dataset import *

dataset_register_table = {
    'ReDial': ReDialDataset,
    'TGReDial': TGReDialDataset,
    'GoRecDial': GoRecDialDataset,
    'OpenDialKG': OpenDialKGDataset,
    'Inspired': InspiredDataset,
    'DuRecDial': DuRecDialDataset
}

dataset_language_map = {
    'ReDial': 'en',
    'TGReDial': 'zh',
    'GoRecDial': 'en',
    'OpenDialKG': 'en',
    'Inspired': 'en',
    'DuRecDial': 'zh'
}

dataloader_register_table = {
    'KGSF': KGSFDataLoader,
    'KBRD': KBRDDataLoader,
    'TGReDial': TGReDialDataLoader,
    'TGRec': TGReDialDataLoader,
    'TGConv': TGReDialDataLoader,
    'TGPolicy': TGReDialDataLoader,
    'TGRec_TGConv': TGReDialDataLoader,
    'TGRec_TGConv_TGPolicy': TGReDialDataLoader,
    'ReDialRec': ReDialDataLoader,
    'ReDialConv': ReDialDataLoader,
    'ReDialRec_ReDialConv': ReDialDataLoader,
    'InspiredRec_InspiredConv': InspiredDataLoader,
    'BERT': TGReDialDataLoader,
    'SASREC': TGReDialDataLoader,
    'TextCNN': TGReDialDataLoader,
    'GRU4REC': TGReDialDataLoader,
    'Popularity': TGReDialDataLoader,
    'Transformer': KGSFDataLoader,
    'GPT2': TGReDialDataLoader,
    'ConvBERT': TGReDialDataLoader,
    'TopicBERT': TGReDialDataLoader,
    'ProfileBERT': TGReDialDataLoader,
    'MGCG': TGReDialDataLoader,
    'PMI': TGReDialDataLoader,
    'NTRD': NTRDDataLoader
}


def get_dataset(opt, tokenize, restore, save) -> BaseDataset:
    """get and process dataset

    Args:
        opt (Config or dict): config for dataset or the whole system.
        tokenize (str): how to tokenize the dataset.
        restore (bool): whether to restore saved dataset which has been processed.
        save (bool): whether to save dataset after processing.

    Returns:
        processed dataset

    """
    dataset = opt['dataset']
    if dataset in dataset_register_table:
        return dataset_register_table[dataset](opt, tokenize, restore, save)
    else:
        raise NotImplementedError(f'The dataloader [{dataset}] has not been implemented')


def get_dataloader(opt, dataset, vocab) -> BaseDataLoader:
    """get dataloader to batchify dataset

    Args:
        opt (Config or dict): config for dataloader or the whole system.
        dataset: processed raw data, no side data.
        vocab (dict): all kinds of useful size, idx and map between token and idx.

    Returns:
        dataloader

    """
    model_name = opt['model_name']
    if model_name in dataloader_register_table:
        return dataloader_register_table[model_name](opt, dataset, vocab)
    else:
        raise NotImplementedError(f'The dataloader [{model_name}] has not been implemented')
