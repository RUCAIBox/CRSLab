# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/24, 2020/12/18
# @Author : Kun Zhou, Xiaolei Wang
# @Email  : francis_kun_zhou@163.com, wxl1999@foxmail.com

import argparse
import warnings

from crslab.config import Config
from crslab.data import get_dataset, get_dataloader
from crslab.system import get_system

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,
                        default='config/kbrd/tgredial.yaml', help='config file(yaml) path')
    parser.add_argument('-sd', '--save_data', action='store_true',
                        help='save processed dataset')
    parser.add_argument('-rd', '--restore_data', action='store_true',
                        help='restore processed dataset')
    parser.add_argument('-ss', '--save_system', action='store_true',
                        help='save trained system')
    parser.add_argument('-rs', '--restore_system', action='store_true',
                        help='restore trained system')
    parser.add_argument('-d', '--debug', action='store_true',
                        help='use valid dataset to debug your system')
    args, _ = parser.parse_known_args()
    opt = Config(args.config, args.debug)
    # dataset and dataloader
    if isinstance(opt['tokenize'], str):
        CRS_dataset = get_dataset(opt, opt['tokenize'], args.restore_data, args.save_data)
        side_data = CRS_dataset.side_data
        vocab = CRS_dataset.vocab

        train_dataloader = get_dataloader(opt, CRS_dataset.train_data, vocab)
        valid_dataloader = get_dataloader(opt, CRS_dataset.valid_data, vocab)
        test_dataloader = get_dataloader(opt, CRS_dataset.test_data, vocab)
    else:
        tokenized_dataset = {}
        train_dataloader = {}
        valid_dataloader = {}
        test_dataloader = {}
        vocab = {}
        side_data = {}

        for task, tokenize in opt['tokenize'].items():
            if tokenize in tokenized_dataset:
                dataset = tokenized_dataset[tokenize]
            else:
                dataset = get_dataset(opt, tokenize, args.restore_data, args.save_data)
                tokenized_dataset[tokenize] = dataset
            train_data = dataset.train_data
            valid_data = dataset.valid_data
            test_data = dataset.test_data
            side_data[task] = dataset.side_data
            vocab[task] = dataset.vocab

            train_dataloader[task] = get_dataloader(opt, train_data, vocab[task])
            valid_dataloader[task] = get_dataloader(opt, valid_data, vocab[task])
            test_dataloader[task] = get_dataloader(opt, test_data, vocab[task])
    # system init, fit and save
    CRS = get_system(opt, train_dataloader, valid_dataloader, test_dataloader, vocab, side_data, args.restore_system,
                     args.debug)
    CRS.fit()
    if args.save_system:
        CRS.save_model()
