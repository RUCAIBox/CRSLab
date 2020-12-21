# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/23, 2020/12/20
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
SAVE_PATH = os.path.join(ROOT_PATH, 'save')
DATA_PATH = os.path.join(ROOT_PATH, 'data')
DATASET_PATH = os.path.join(DATA_PATH, 'dataset')
MODEL_PATH = os.path.join(DATA_PATH, 'model')
EMBEDDING_PATH = os.path.join(DATA_PATH, 'embedding')


class Config:
    def __init__(self, config_file=None, debug=False):
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
