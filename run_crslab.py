# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/24, 2020/11/29
# @Author : Kun Zhou, Xiaolei Wang
# @Email  : francis_kun_zhou@163.com, wxl1999@foxmail.com

import argparse
from pprint import pprint

from crslab.config import Config
from crslab.data import get_dataset, get_dataloader
from crslab.system import get_system

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default='config/kgsf/redial.yaml', help='config files')
    parser.add_argument('--save', type=bool, default=False, help='whether save the preprocessed dataset')
    parser.add_argument('--restore', type=bool, default=False, help='whether restore the preprocessed dataset')

    args, _ = parser.parse_known_args()

    config = Config(config_file=args.config_file)

    # dataset splitting
    CRS_dataset = get_dataset(config, args.restore, args.save)
    train_data = CRS_dataset.train_data
    valid_data = CRS_dataset.valid_data
    test_data = CRS_dataset.test_data
    side_data = CRS_dataset.side_data
    ind2token = CRS_dataset.ind2tok

    for i, data in enumerate(train_data):
        pprint(data)
        break

    exit()

    train_dataloader = get_dataloader(config, train_data)
    valid_dataloader = get_dataloader(config, valid_data)
    test_dataloader = get_dataloader(config, test_data)
    '''
    conv_data=valid_dataloader.get_conv_data(config['batch_size']['conv'], )
    pretrain_data=valid_dataloader.get_conv_data(config['batch_size']['rec'])
    rec_data=valid_dataloader.get_conv_data(config['batch_size']['rec'])
    rgcn_data=valid_dataloader.get_side_data(side_data[0], 'RGCN')
    gcn_data=valid_dataloader.get_side_data(side_data[1], 'GCN')
    '''

    # system loading and initialization
    CRS = get_system(config, train_dataloader, valid_dataloader, test_dataloader, ind2token, side_data)
    CRS.fit(debug=True)
