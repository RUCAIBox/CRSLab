from loguru import logger

from .conv_evaluator import ConvEvaluator
from .rec_evaluator import RecEvaluator
from .standard_evaluator import StandardEvaluator

Evaluator_register_table = {
    'rec': RecEvaluator,
    'conv': ConvEvaluator,
    'standard': StandardEvaluator
}

dataset_language_map = {
    'ReDial': 'en',
    'TGReDial': 'zh',
    'GoRecDial': 'en',
    'OpenDialKG': 'en',
    'Inspired': 'en',
}


def get_evaluator(evaluator_name, dataset):
    if evaluator_name in Evaluator_register_table:
        if evaluator_name in ('conv', 'standard'):
            language = dataset_language_map[dataset]
            evaluator = Evaluator_register_table[evaluator_name](language)
        else:
            evaluator = Evaluator_register_table[evaluator_name]()
        logger.info(f'[Build evaluator {evaluator_name}]')
        return evaluator
    else:
        raise NotImplementedError(f'Model [{evaluator_name}] has not been implemented')
