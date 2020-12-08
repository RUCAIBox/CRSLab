# @Time   : 2020/11/30
# @Author : Xiaolei Wang
# @Email  : wxl1999@foxmail.com

# UPDATE:
# @Time   : 2020/11/30
# @Author : Xiaolei Wang
# @Email  : wxl1999@foxmail.com

from abc import ABC, abstractmethod

from crslab.evaluator.metrics.base_metrics import Metrics


class BaseEvaluator(ABC):
    def __init__(self):
        self.optim_metrics = Metrics()

    def rec_evaluate(self, preds, label):
        pass

    def gen_evaluate(self, preds, label):
        pass

    def policy_evaluate(self, preds, label):
        pass

    @abstractmethod
    def report(self):
        pass

    @abstractmethod
    def reset_metrics(self):
        self.optim_metrics.clear()
