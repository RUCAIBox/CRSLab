# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/24, 2020/12/29, 2020/12/17, 2021/8/4
# @Author : Kun Zhou, Xiaolei Wang, Yuanhang Zhou, Chenzhan Shang
# @Email  : francis_kun_zhou@163.com, wxl1999@foxmail.com, sdzyh002@gmail.com, czshang@outlook.com

from .base import TextBaseDataset
from .durecdial import DuRecDialDataset
from .gorecdial import GoRecDialDataset
from .inspired import InspiredDataset
from .opendialkg import OpenDialKGDataset
from .redial import ReDialDataset
from .tgredial import TGReDialDataset
from .lastfm import LastFMDataset

dataset_register_table = {
    'ReDial': ReDialDataset,
    'TGReDial': TGReDialDataset,
    'GoRecDial': GoRecDialDataset,
    'OpenDialKG': OpenDialKGDataset,
    'Inspired': InspiredDataset,
    'DuRecDial': DuRecDialDataset,
    'LastFM': LastFMDataset
}

dataset_language_map = {
    'ReDial': 'en',
    'TGReDial': 'zh',
    'GoRecDial': 'en',
    'OpenDialKG': 'en',
    'Inspired': 'en',
    'DuRecDial': 'zh'
}


def get_dataset(opt, tokenize, restore, save):
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
        raise NotImplementedError(f'The supervised [{dataset}] has not been implemented')

