# -*- encoding: utf-8 -*-
# @Time    :   2020/12/22
# @Author  :   Xiaolei Wang
# @email   :   wxl1999@foxmail.com

# UPDATE
# @Time    :   2020/12/22, 2021/8/4
# @Author  :   Xiaolei Wang, Chenzhan Shang
# @email   :   wxl1999@foxmail.com, czshang@outlook.com

from loguru import logger

from crslab.evaluator.conv import ConvEvaluator
from crslab.evaluator.rec import RecEvaluator
from crslab.evaluator.standard import StandardEvaluator
from crslab.dataset import dataset_language_map

evaluator_register_table = {
    'rec': RecEvaluator,
    'conv': ConvEvaluator,
    'standard': StandardEvaluator
}


def get_evaluator(evaluator_name, dataset, tensorboard=False):
    if evaluator_name in evaluator_register_table:
        if evaluator_name in ('conv', 'standard'):
            language = dataset_language_map[dataset]
            evaluator = evaluator_register_table[evaluator_name](language, tensorboard=tensorboard)
        else:
            evaluator = evaluator_register_table[evaluator_name](tensorboard=tensorboard)
        logger.info(f'[Build evaluator {evaluator_name}]')
        return evaluator
    else:
        raise NotImplementedError(f'Model [{evaluator_name}] has not been implemented')
