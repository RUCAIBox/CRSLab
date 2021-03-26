# -*- encoding: utf-8 -*-
# @Time    :   2021/1/8
# @Author  :   Xiaolei Wang
# @email   :   wxl1999@foxmail.com

# UPDATE
# @Time    :   2021/1/9
# @Author  :   Xiaolei Wang
# @email   :   wxl1999@foxmail.com

from crslab.config import Config
from crslab.data import get_dataset, get_dataloader
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

    .. _Github repo:
       https://github.com/RUCAIBox/CRSLab

    """
    # dataset & dataloader
    if isinstance(config['tokenize'], str):
        CRS_dataset = get_dataset(config, config['tokenize'], restore_data, save_data)
        side_data = CRS_dataset.side_data
        vocab = CRS_dataset.vocab

        train_dataloader = get_dataloader(config, CRS_dataset.train_data, vocab)
        valid_dataloader = get_dataloader(config, CRS_dataset.valid_data, vocab)
        test_dataloader = get_dataloader(config, CRS_dataset.test_data, vocab)
    else:
        tokenized_dataset = {}
        train_dataloader = {}
        valid_dataloader = {}
        test_dataloader = {}
        vocab = {}
        side_data = {}

        for task, tokenize in config['tokenize'].items():
            if tokenize in tokenized_dataset:
                dataset = tokenized_dataset[tokenize]
            else:
                dataset = get_dataset(config, tokenize, restore_data, save_data)
                tokenized_dataset[tokenize] = dataset
            train_data = dataset.train_data
            valid_data = dataset.valid_data
            test_data = dataset.test_data
            side_data[task] = dataset.side_data
            vocab[task] = dataset.vocab

            train_dataloader[task] = get_dataloader(config, train_data, vocab[task])
            valid_dataloader[task] = get_dataloader(config, valid_data, vocab[task])
            test_dataloader[task] = get_dataloader(config, test_data, vocab[task])
    # system
    CRS = get_system(config, train_dataloader, valid_dataloader, test_dataloader, vocab, side_data, restore_system,
                     interact, debug, tensorboard)
    if interact:
        CRS.interact()
    else:
        CRS.fit()
        if save_system:
            CRS.save_model()
