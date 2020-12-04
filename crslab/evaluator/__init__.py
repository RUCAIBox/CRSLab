from loguru import logger

from .standard_evaluator import StandardEvaluator
from .rec_evaluator import RecEvaluator
from .conv_evaluator import ConvEvaluator


Evaluator_register_table = {
    'rec': RecEvaluator,
    'conv': ConvEvaluator,
    'standard': StandardEvaluator
}


def get_evaluator(evaluator_name):
    if evaluator_name in Evaluator_register_table:
        evaluator = Evaluator_register_table[evaluator_name]()
        logger.info(f'[Build evaluator {evaluator_name}]')
        return evaluator
    else:
        raise NotImplementedError(f'Model [{evaluator_name}] has not been implemented')
