# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/23
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

import json
import os
import time
from pprint import pprint

import yaml
from loguru import logger
from tqdm import tqdm


class Config(object):
    """ Configurator module that load the defined parameters.

    Configurator module will first load the default parameters from the fixed properties in RecBole and then
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
        self.config_dict = self._load_yaml_configs(config_file)

        if not os.path.exists("log"):
            os.makedirs("log")
        log_name = self.config_dict.get("log_name",
                                        self.config_dict['dataset'] + '_' + self.config_dict['rec_model'] + '_' +
                                        self.config_dict['conv_model'] + '_' + time.strftime("%Y-%m-%d-%H-%M-%S",
                                                                                             time.localtime())) + ".log"
        logger.remove()
        logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
        logger.add(os.path.join("log", log_name))

        logger.info("[DATASET: {}]", self.config_dict['dataset'])
        logger.info("[Recommender Model: {}]", self.config_dict['rec_model'])
        logger.info("[Conversation Model: {}]", self.config_dict['conv_model'])
        logger.info("[CONFIG]" + '\n' + json.dumps(self.config_dict, indent=4))

        # self.model, self.model_class, self.dataset = self._get_model_and_dataset(self.config_dict['dataset'],
        #                                                  self.config_dict['rec_model'],self.config_dict['conv_model'])
        # self._init_device()

    def _load_yaml_configs(self, filename):
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
        self.config_dict[key] = value

    def __getitem__(self, item):
        if item in self.config_dict:
            return self.config_dict[item]
        else:
            return None

    def get(self, item, default=None):
        if item in self.config_dict:
            return self.config_dict[item]
        else:
            return default

    def __contains__(self, key):
        if not isinstance(key, str):
            raise TypeError("index must be a str.")
        return key in self.config_dict

    def __repr__(self):
        return self.__str__()


if __name__ == '__main__':
    opt_dict = Config('../../properties/kbrd_redial.yaml')
    pprint(opt_dict)
