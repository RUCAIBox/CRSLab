# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/24, 2021/1/9
# @Author : Kun Zhou, Xiaolei Wang
# @Email  : francis_kun_zhou@163.com, wxl1999@foxmail.com

import argparse
import warnings

from crslab.config import Config

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,
                        default='config/crs/tgredial/tgredial.yaml', help='config file(yaml) path')
    parser.add_argument('-g', '--gpu', type=str, default='-1',
                        help='specify GPU id(s) to use, we now support multiple GPUs. Defaults to CPU(-1).')
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
    parser.add_argument('-i', '--interact', action='store_true',
                        help='interact with your system instead of training')
    parser.add_argument('-tb', '--tensorboard', action='store_true',
                        help='enable tensorboard to monitor train performance')
    parser.add_argument('--mode', choices=['train', 'chat', 'ask'],
                        help='Train CRS model / Evaluating the CRS model with iEvaLM free-form chit-chat mode / attribute-based question answering mode')
    args, _ = parser.parse_known_args()
    config = Config(args.config, args.gpu, args.debug)

    from crslab.quick_start import quick_start
    
    quick_start(config, args.mode, args.save_data, args.restore_data, args.save_system, args.restore_system, args.interact,
               args.debug, args.tensorboard)
