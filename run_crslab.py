# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/24, 2020/12/2
# @Author : Kun Zhou, Xiaolei Wang
# @Email  : francis_kun_zhou@163.com, wxl1999@foxmail.com

import argparse

from crslab.config import Config
from crslab.data import get_dataset, get_dataloader
from crslab.system import get_system


if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,
                        default='config/kbrd/redial.yaml', help='config file(yaml) path')
    parser.add_argument('-s', '--save', action='store_true',
                        help='save processed dataset and model')
    parser.add_argument('-r', '--restore', action='store_true',
                        help='restore processed dataset and model')
    parser.add_argument('-d', '--debug', action='store_true',
                        help='use valid dataset to debug your system')
    args, _ = parser.parse_known_args()
    config = Config(args.config, args.debug)
    # dataset
    CRS_dataset = get_dataset(config, args.restore, args.save)
    train_data = CRS_dataset.train_data
    valid_data = CRS_dataset.valid_data
    test_data = CRS_dataset.test_data
    side_data = CRS_dataset.side_data
    ind2token = CRS_dataset.ind2tok
    # dataloader
    train_dataloader = get_dataloader(config, train_data)
    valid_dataloader = get_dataloader(config, valid_data)
    test_dataloader = get_dataloader(config, test_data)
    # system init and fit
    CRS = get_system(config, train_dataloader, valid_dataloader, test_dataloader, ind2token, side_data, args.restore,
                     args.save, args.debug)
    CRS.fit()
