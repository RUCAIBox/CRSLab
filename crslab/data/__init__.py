# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/24, 2020/12/22, 2020/12/17
# @Author : Kun Zhou, Xiaolei Wang, Yuanhang Zhou
# @Email  : francis_kun_zhou@163.com, wxl1999@foxmail.com, sdzyh002@gmail.com

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
}


def get_dataset(opt, tokenize, restore, save) -> BaseDataset:
    dataset = opt['dataset']
    if dataset in dataset_register_table:
        return dataset_register_table[dataset](opt, tokenize, restore, save)
    else:
        raise NotImplementedError(f'The dataloader [{dataset}] has not been implemented')


def get_dataloader(opt, dataset, vocab) -> BaseDataLoader:
    model_name = opt['model_name']
    if model_name in dataloader_register_table:
        return dataloader_register_table[model_name](opt, dataset, vocab)
    else:
        raise NotImplementedError(f'The dataloader [{model_name}] has not been implemented')
