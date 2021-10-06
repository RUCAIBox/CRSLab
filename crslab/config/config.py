# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/23, 2021/1/9
# @Author : Kun Zhou, Xiaolei Wang
# @Email  : francis_kun_zhou@163.com, wxl1999@foxmail.com

import json
import os
import time
import sys
import traceback
from pprint import pprint

import torch
import numpy as np
import random
import yaml
from loguru import logger
from tqdm import tqdm


class Config:
    """Configurator module that load the defined parameters and initialize the whole system."""

    def __init__(self, args):
        """Load parameters and set log level.

        Args:
            args (namespace): arguments provided by command line, which specifies dataset, model,
            external config files and etc.

        """
        external_config_files = args.config.strip().split(' ') if args.config else None
        self.external_file_config_dict = self._load_config_files(external_config_files)
        self.cmd_config_dict = self._load_cmd_line()
        self._merge_external_config_dict()
        self.model, self.dataset = self._get_model_and_dataset_name(args)
        self._load_internal_config_dict(self.model, self.dataset)
        self.final_config_dict = self._get_final_config_dict()

        self._init_device()
        self._init_random_seed()
        self._init_paths()
        self._init_logger()

    @staticmethod
    def _load_config_files(config_files):
        config_dict = dict()
        if config_files:
            for file in config_files:
                with open(file, 'r', encoding='utf-8') as f:
                    config_dict.update(yaml.safe_load(f.read()))
        return config_dict

    def _deep_update(self, target_dict, source_dict):
        for source_key in source_dict:
            if source_key not in target_dict.keys() or not isinstance(source_dict[source_key], dict):
                target_dict[source_key] = source_dict[source_key]
            else:
                assert isinstance(target_dict[source_key], dict)
                self._deep_update(target_dict[source_key], source_dict[source_key])

    def _convert_config_dict(self, raw_config_dict):
        config_dict = dict()
        for key in raw_config_dict:
            param = raw_config_dict[key]
            if not isinstance(param, str):
                continue
            try:
                value = eval(param)
                if not isinstance(value, (str, int, float, list, tuple, dict, bool)):
                    value = param
            except (NameError, SyntaxError, TypeError):
                if isinstance(param, str):
                    if param.lower() == "true":
                        value = True
                    elif param.lower() == "false":
                        value = False
                    else:
                        value = param
                else:
                    value = param
            param_list = key.split(".")[::-1]
            param_dict = value
            for i in range(len(param_list)):
                param_dict = {param_list[i]: param_dict}
            self._deep_update(config_dict, param_dict)
        return config_dict

    def _load_cmd_line(self):
        config_dict = dict()
        unrecognized_args = []
        for arg in sys.argv[1:]:
            if not arg.startswith("--") or len(arg[2:].split("=")) != 2:
                unrecognized_args.append(arg)
                continue
            arg_name, arg_value = arg[2:].split("=")
            if arg_name in config_dict and arg_value != config_dict[arg_name]:
                raise SyntaxError("There are duplicate commend arg '%s' with different value." % arg)
            else:
                config_dict[arg_name] = arg_value
        if len(unrecognized_args) > 0:
            logger.warning(f"[Command line args {' '.join(unrecognized_args)} will not be used in CRSLab.]")
        config_dict = self._convert_config_dict(config_dict)
        return config_dict

    def _merge_external_config_dict(self):
        external_config_dict = dict()
        self._deep_update(external_config_dict, self.external_file_config_dict)
        self._deep_update(external_config_dict, self.cmd_config_dict)
        self.external_config_dict = external_config_dict

    def _get_model_and_dataset_name(self, args):
        model = args.model
        if model is None:
            try:
                model = self.external_config_dict['model']
            except KeyError:
                raise KeyError(
                    'model need to be specified in at least one of the these ways: '
                    '[config file, command line] '
                )

        dataset = args.dataset
        if dataset is None:
            try:
                dataset = self.external_config_dict['dataset']
            except KeyError:
                raise KeyError(
                    'dataset need to be specified in at least one of the these ways: '
                    '[config file, command line] '
                )
        return model, dataset

    def _load_internal_config_dict(self, model, dataset):
        self.internal_config_dict = dict()
        current_path = os.path.dirname(os.path.realpath(__file__))
        overall_config_file = os.path.join(current_path, '../../config/overall.yaml')
        model_config_file = os.path.join(current_path, '../../config/model/' + model.lower() + '.yaml')
        optim_config_file = os.path.join(current_path, '../../config/optim/' + model.lower() + '.yaml')
        dataset_config_file = os.path.join(current_path, '../../config/dataset/' + dataset.lower() + '.yaml')

        for file in [overall_config_file, model_config_file, optim_config_file, dataset_config_file]:
            if os.path.isfile(file):
                with open(file, 'r', encoding='utf-8') as f:
                    config_dict = yaml.safe_load(f.read())
                    if config_dict is not None:
                        self.internal_config_dict.update(config_dict)

    def _get_final_config_dict(self):
        final_config_dict = dict()
        self._deep_update(final_config_dict, self.internal_config_dict)
        self._deep_update(final_config_dict, self.external_config_dict)
        return final_config_dict

    def _init_device(self):
        use_gpu = self.final_config_dict['use_gpu']
        if use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.final_config_dict['gpu_id'])
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")

    def _init_random_seed(self):
        seed = self.final_config_dict.get('seed', 2021)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _init_paths(self):
        # cache paths which are transparent to users
        self.cache_home = os.path.expanduser(os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), 'crslab'))
        self.data_path = os.path.join(self.cache_home, 'data')
        self.dataset_path = os.path.join(self.data_path, 'dataset')
        self.model_path = os.path.join(self.data_path, 'model')
        self.pretrain_path = os.path.join(self.model_path, 'pretrain')
        self.embedding_path = os.path.join(self.data_path, 'embedding')

        # user-defined paths
        stack = traceback.extract_stack()
        self.root_path = os.path.expanduser(os.path.dirname(os.path.realpath(stack[-3].filename)))
        self.log_path = os.path.join(self.root_path, self.final_config_dict.get('log_path', 'log'))
        self.checkpoint_path = os.path.join(self.root_path, self.final_config_dict.get('checkpoint_path', 'checkpoint'))
        self.intermediate_path = os.path.join(self.root_path, self.final_config_dict.get('intermediate_path', 'data'))
        self.customized_path = os.path.join(self.root_path, self.final_config_dict.get('customized_path', 'custom'))

    def _init_logger(self):
        log_name = self.model + '_' + self.dataset + '_' + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + '.log'
        log_name = self.final_config_dict.get('log_name', log_name)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        logger.remove()
        level = 'DEBUG' if self.final_config_dict.get('debug', False) else 'INFO'
        logger.add(os.path.join(self.log_path, log_name), level=level)
        logger.add(lambda msg: tqdm.write(msg, end=''), colorize=True, level=level)
        logger.info(f"[Dataset: {self.dataset} tokenized in {self.final_config_dict['tokenize']}]")
        logger.info(f"[Model: {self.model}]")
        logger.info("[Config]\n" + json.dumps(self.final_config_dict, indent=4))

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise TypeError("index must be a str.")
        self.final_config_dict[key] = value

    def __getitem__(self, item):
        if item in self.final_config_dict:
            return self.final_config_dict[item]
        else:
            return None

    def __contains__(self, key):
        if not isinstance(key, str):
            raise TypeError("index must be a str.")
        return key in self.final_config_dict

    def __str__(self):
        return str(self.final_config_dict)

    def __repr__(self):
        return self.__str__()

    def get(self, item, default=None):
        """Get value of corrsponding item in config

        Args:
            item (str): key to query in config
            default (optional): default value for item if not found in config. Defaults to None.

        Returns:
            value of corrsponding item in config

        """
        if item in self.final_config_dict:
            return self.final_config_dict[item]
        else:
            return default


if __name__ == '__main__':
    opt_dict = Config('../../config/generation/kbrd/redial.yaml')
    pprint(opt_dict)
