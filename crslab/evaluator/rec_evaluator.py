# @Time   : 2020/11/30
# @Author : Xiaolei Wang
# @Email  : wxl1999@foxmail.com

# UPDATE:
# @Time   : 2020/11/30
# @Author : Xiaolei Wang
# @Email  : wxl1999@foxmail.com

from loguru import logger

from crslab.evaluator.base_evaluator import BaseEvaluator
from crslab.evaluator.metrics import Metrics, aggregate_unnamed_reports
from crslab.evaluator.rec_metrics import RecallMetric
from crslab.system.utils import nice_report


class RecEvaluator(BaseEvaluator):
    def __init__(self):
        super(RecEvaluator, self).__init__()
        self.rec_metrics = Metrics()

    def evaluate(self, preds, label):
        for k in [1, 10, 50]:
            if len(preds) >= k:
                self.rec_metrics.add(f"recall@{k}", RecallMetric.compute(preds, label, k))

    def report(self):
        reports = [self.rec_metrics.report(), self.optim_metrics.report()]
        logger.info('\n' + nice_report(aggregate_unnamed_reports(reports)))

    def reset_metrics(self):
        super(RecEvaluator, self).reset_metrics()
        self.rec_metrics.clear()
