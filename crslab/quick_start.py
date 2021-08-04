# -*- encoding: utf-8 -*-
# @Time    :   2021/1/8
# @Author  :   Xiaolei Wang
# @email   :   wxl1999@foxmail.com

# UPDATE
# @Time    :   2021/1/9, 2021/8/4
# @Author  :   Xiaolei Wang, Chenzhan Shang
# @email   :   wxl1999@foxmail.com, czshang@outlook.com

from crslab.config import Config
from crslab.agent import get_agent
from crslab.dataset import get_dataset
from crslab.system import get_system


def run_crslab(config, save_data=False, restore_data=False, save_system=False, restore_system=False,
               interact=False, debug=False, tensorboard=False):
    """A fast running api, which includes the complete process of training and testing models on specified datasets.

    Args:
        config (Config or str): an instance of ``Config`` or path to the config file,
            which should be in ``yaml`` format. You can use default config provided in the `Github repo`_,
            or write it by yourself.
        save_data (bool): whether to save data. Defaults to False.
        restore_data (bool): whether to restore data. Defaults to False.
        save_system (bool): whether to save system. Defaults to False.
        restore_system (bool): whether to restore system. Defaults to False.
        interact (bool): whether to interact with the system. Defaults to False.
        debug (bool): whether to debug the system. Defaults to False.
        tensorboard (bool): whether to use tensorboard. Defaults to False.

    .. _Github repo:
       https://github.com/RUCAIBox/CRSLab

    """
    dataset = get_dataset(config, config['tokenize'], restore_data, save_data)
    agent = get_agent(config, dataset)
    system = get_system(config, agent, restore_system, interact, debug, tensorboard)
    if interact:
        # interact with user
        system.interact()
    else:
        system.run()
        if save_system:
            system.save_model()
