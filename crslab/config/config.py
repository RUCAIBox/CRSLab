# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/23, 2020/12/13
# @Author : Kun Zhou, Xiaolei Wang
# @Email  : francis_kun_zhou@163.com, wxl1999@foxmail.com

import json
import os
import time
from os.path import dirname, realpath
from pprint import pprint

import yaml
from loguru import logger
from tqdm import tqdm

ROOT_PATH = dirname(dirname(dirname(realpath(__file__))))
DATA_PATH = os.path.join(ROOT_PATH, 'data')
DATASET_PATH = os.path.join(DATA_PATH, 'dataset')
MODEL_PATH = os.path.join(DATA_PATH, 'model')
SAVE_PATH = os.path.join(ROOT_PATH, 'save')


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

    def __init__(self, config_file=None, debug=False):
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
        tokenize = self.opt['tokenize']
        if isinstance(tokenize, dict):
            tokenize = ', '.join(tokenize.values())
        model = self.opt.get('model', None)
        rec_model = self.opt.get('rec_model', None)
        conv_model = self.opt.get('conv_model', None)
        policy_model = self.opt.get('policy_model', None)
        if model:
            model_name = model
        else:
            models = []
            if rec_model:
                models.append(rec_model)
            if conv_model:
                models.append(conv_model)
            if policy_model:
                models.append(policy_model)
            model_name = '_'.join(models)
        self.opt['model_name'] = model_name

        log_name = self.opt.get("log_name", dataset + '_' + model_name + '_' + time.strftime("%Y-%m-%d-%H-%M-%S",
                                                                                             time.localtime())) + ".log"
        if not os.path.exists("log"):
            os.makedirs("log")
        logger.remove()
        if debug:
            level = 'DEBUG'
        else:
            level = 'INFO'
        logger.add(os.path.join("log", log_name), level=level)
        logger.add(lambda msg: tqdm.write(msg, end=''), colorize=True, level=level)

        logger.info(f"[Dataset: {dataset} tokenized in {tokenize}]")
        if model:
            logger.info(f'[Model: {model}]')
        if rec_model:
            logger.info(f'[Recommendation Model: {rec_model}]')
        if conv_model:
            logger.info(f'[Conversation Model: {conv_model}]')
        if policy_model:
            logger.info(f'[Policy Model: {policy_model}]')
        logger.info("[Config]" + '\n' + json.dumps(self.opt, indent=4))

    @staticmethod
    def load_yaml_configs(filename):
        """
        This function reads yaml file to build config dictionary
        """
        config_dict = dict()
        with open(filename, 'r', encoding='utf-8') as f:
            config_dict.update(yaml.safe_load(f.read()))
        return config_dict

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
