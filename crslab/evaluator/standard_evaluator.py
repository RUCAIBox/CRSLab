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


class StandardEvaluator(BaseEvaluator):
    def __init__(self):
        super(StandardEvaluator, self).__init__()
        # rec
        self.rec_metrics = Metrics()
        # gen
        self.dist_set = defaultdict(set)
        self.dist_cnt = 0
        self.gen_metrics = Metrics()
        # optim
        self.optim_metrics = Metrics()

    def rec_evaluate(self, ranks, label):
        for k in [1, 10, 50]:
            if len(ranks) >= k:
                self.rec_metrics.add(f"hit@{k}", HitMetric.compute(ranks, label, k))
                self.rec_metrics.add(f"ndcg@{k}", NDCGMetric.compute(ranks, label, k))
                self.rec_metrics.add(f"mrr@{k}", MRRMetric.compute(ranks, label, k))

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
        reports = [self.rec_metrics.report(), self.gen_metrics.report(), self.optim_metrics.report()]
        logger.info('\n' + nice_report(aggregate_unnamed_reports(reports)))

    def reset_metrics(self):
        # rec
        self.rec_metrics.clear()
        # conv
        self.gen_metrics.clear()
        self.dist_cnt = 0
        self.dist_set.clear()
        # optim
        self.optim_metrics.clear()
