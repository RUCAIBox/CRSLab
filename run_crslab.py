# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/24, 2021/1/9, 2021/8/4
# @Author : Kun Zhou, Xiaolei Wang, Chenzhan Shang
# @Email  : francis_kun_zhou@163.com, wxl1999@foxmail.com, czshang@outlook.com

import argparse
import warnings

from crslab.config import Config
from crslab.agent import get_agent
from crslab.dataset import get_dataset
from crslab.system import get_system

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='ReDial', help='name of dataset')
    parser.add_argument('-m', '--model', type=str, default='KBRD', help='name of model')
    parser.add_argument('-c', '--config', type=str, default=None, help='external config files')
    parser.add_argument('-rd', '--restore_data', action='store_true', help='restore processed dataset')
    parser.add_argument('-rm', '--restore_model', action='store_true', help='restore trained model')
    parser.add_argument('-i', '--interaction', action='store_true', help='interact with your system')
    args, _ = parser.parse_known_args()
    config = Config(args)

    dataset = get_dataset(config, config['tokenize'])
    agent = get_agent(config, dataset)
    system = get_system(config, agent)
    system.run()
