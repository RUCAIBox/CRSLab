# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/24, 2020/11/30
# @Author : Kun Zhou, Xiaolei Wang
# @Email  : francis_kun_zhou@163.com, wxl1999@foxmail.com

import argparse
from pprint import pprint

from crslab.config import Config
from crslab.data import get_dataset, get_dataloader
from crslab.system import get_system

if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default='config/kbrd/redial.yaml', help='config files')
    parser.add_argument('--save', type=bool, default=False, help='whether save the preprocessed dataset')
    parser.add_argument('--restore', type=bool, default=False, help='whether restore the preprocessed dataset')
    args, _ = parser.parse_known_args()
    config = Config(config_file=args.config_file)
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

    for i, batch in enumerate(train_dataloader.get_conv_data(32, shuffle=True)):
        pprint(batch)
        if i == 5:
            break
    exit()
    # system init and fit
    CRS = get_system(config, train_dataloader, valid_dataloader, test_dataloader, ind2token, side_data)
    CRS.fit(debug=True)
