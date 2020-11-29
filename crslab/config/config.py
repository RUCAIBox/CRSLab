# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/23, 2020/11/29
# @Author : Kun Zhou, Xiaolei Wang
# @Email  : francis_kun_zhou@163.com, wxl1999@foxmail.com

import json
import os
import time
from pprint import pprint

import yaml
from loguru import logger
from tqdm import tqdm


class Config:
    """ Configurator module that load the defined parameters.

    Configurator module will first load the default parameters from the fixed config in RecBole and then
    load parameters from the external input.

    External input supports three kind of forms: config file, command line and parameter dictionaries.

    - config file: It's a file that record the parameters to be modified or added. It should be in ``yaml`` format,
      e.g. a config file is 'example.yaml', the content is:

        learning_rate: 0.001

        train_batch_size: 2048

    - command line: It should be in the format as '--learning_rate=0.001'

    - parameter dictionaries: It should be a dict, where the key is parameter name and the value is parameter value,
      e.g. config_dict = {'learning_rate': 0.001}

    Configuration module allows the above three kind of external input format to be used together,
    the priority order is as following:

    command line > parameter dictionaries > config file

    e.g. If we set learning_rate=0.01 in config file, learning_rate=0.02 in command line,
    learning_rate=0.03 in parameter dictionaries.

    Finally the learning_rate is equal to 0.02.
    """

    def __init__(self, config_file=None):
        """
        Args:
            model (str/AbstractRecommender): the model name or the model class, default is None, if it is None, config
            will search the parameter 'model' from the external input as the model name or model class.
            dataset (str): the dataset name, default is None, if it is None, config will search the parameter 'dataset'
            from the external input as the dataset name.
            config_file_list (list of str): the external config file, it allows multiple config files, default is None.
            config_dict (dict): the external parameter dictionaries, default is None.
        """
        self.opt = self.load_yaml_configs(config_file)
        dataset = self.opt['dataset']
        rec_model = self.opt.get('rec_model')
        conv_model = self.opt.get('conv_model')

        if not os.path.exists("log"):
            os.makedirs("log")
        logger.remove()
        logger.add(lambda msg: tqdm.write(msg, end=''), colorize=True)
        log_name = self.opt.get("log_name",
                                dataset + '_' + rec_model + '_' + conv_model + '_' + time.strftime("%Y-%m-%d-%H-%M-%S",
                                                                                                   time.localtime())) + ".log"
        logger.add(os.path.join("log", log_name))

        logger.info("[DATASET: {}]", dataset)
        logger.info("[Recommender Model: {}]", rec_model)
        logger.info("[Conversation Model: {}]", conv_model)
        logger.info("[CONFIG]" + '\n' + json.dumps(self.opt, indent=4))

        # self.model, self.model_class, self.dataset = self._get_model_and_dataset(self.config_dict['dataset'],
        #                                                  self.config_dict['rec_model'],self.config_dict['conv_model'])
        # self._init_device()

    @staticmethod
    def load_yaml_configs(filename):
        """
        This function reads yaml file to build config dictionary
        """
        config_dict = dict()
        with open(filename, 'r', encoding='utf-8') as f:
            config_dict.update(yaml.safe_load(f.read()))
        return config_dict

    '''
    def _init_device(self):
        use_gpu = self.config_dict['use_gpu']
        if use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.config_dict['gpu_id'])
        self.config_dict['device'] = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
    '''

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise TypeError("index must be a str.")
        self.opt[key] = value

    def __getitem__(self, item):
        if item in self.opt:
            return self.opt[item]
        else:
            return None

    def get(self, item, default=None):
        if item in self.opt:
            return self.opt[item]
        else:
            return default

    def __contains__(self, key):
        if not isinstance(key, str):
            raise TypeError("index must be a str.")
        return key in self.opt

    def __str__(self):
        return str(self.opt)

    def __repr__(self):
        return self.__str__()


if __name__ == '__main__':
    opt_dict = Config('../../config/kbrd/redial.yaml')
    pprint(opt_dict)
