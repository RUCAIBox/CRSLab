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
    parser.add_argument('-c', '--config', type=str,
                        default='config/crs/tgredial/tgredial.yaml', help='config file(yaml) path')
    parser.add_argument('-g', '--gpu', type=str, default='-1',
                        help='specify GPU id(s) to use, we now support multiple GPUs. Defaults to CPU(-1).')
    parser.add_argument('-rd', '--restore_data', action='store_true',
                        help='restore processed dataset')
    parser.add_argument('-sd', '--save_data', action='store_true',
                        help='save processed dataset')
    parser.add_argument('-rm', '--restore_model', action='store_true',
                        help='restore trained model')
    parser.add_argument('-sm', '--save_model', action='store_true',
                        help='save trained model')
    parser.add_argument('-d', '--debug', action='store_true',
                        help='use valid dataset to debug your system')
    parser.add_argument('-i', '--interaction', action='store_true',
                        help='interact with your system instead of training')
    parser.add_argument('-tb', '--tensorboard', action='store_true',
                        help='enable tensorboard to monitor train performance')
    args, _ = parser.parse_known_args()
    config = Config(args.config, args.gpu, args.debug)

    dataset = get_dataset(config, config['tokenize'], args.restore_data, args.save_data)
    agent = get_agent(config, dataset)
    system = get_system(config, agent, args.restore_model, args.save_model, args.interaction, args.tensorboard)
    system.run()
