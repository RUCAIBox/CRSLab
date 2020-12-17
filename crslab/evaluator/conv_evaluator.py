# @Time   : 2020/11/30
# @Author : Xiaolei Wang
# @Email  : wxl1999@foxmail.com

# UPDATE:
# @Time   : 2020/12/17
# @Author : Xiaolei Wang
# @Email  : wxl1999@foxmail.com

from collections import defaultdict

from loguru import logger
from nltk import ngrams

from crslab.evaluator.base_evaluator import BaseEvaluator
from crslab.evaluator.utils import nice_report
from .metrics import *


class ConvEvaluator(BaseEvaluator):
    def __init__(self):
        super(ConvEvaluator, self).__init__()
        self.dist_set = defaultdict(set)
        self.dist_cnt = 0
        self.gen_metrics = Metrics()
        self.optim_metrics = Metrics()

    def gen_evaluate(self, preds, label):
        if preds:
            self.gen_metrics.add("f1", F1Metric.compute(preds, label))
            for k in range(1, 5):
                self.gen_metrics.add(f"bleu@{k}", BleuMetric.compute(preds, label, k))
                for token in ngrams(preds, k):
                    self.dist_set[f"dist@{k}"].add(token)
            self.dist_cnt += 1

    def report(self):
        for k, v in self.dist_set.items():
            self.gen_metrics.add(k, AverageMetric(len(v) / self.dist_cnt))
        reports = [self.gen_metrics.report(), self.optim_metrics.report()]
        logger.info('\n' + nice_report(aggregate_unnamed_reports(reports)))

    def reset_metrics(self):
        self.gen_metrics.clear()
        self.dist_cnt = 0
        self.dist_set.clear()
        self.optim_metrics.clear()
