# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/24
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

from typing import Optional

from crslab.evaluator.metrics import GenMetrics, RecMetrics, aggregate_unnamed_reports, Metrics, Metric, \
    EvalMetrics

class Evaluator:
    def __init__(self):
        self.all_metrics = {
            "rec": RecMetrics(),
            "conv": GenMetrics(),
            "other": Metrics()
        }

    def add_metric(self, category: str, key: str, value: Optional[Metric]):
        if category in self.all_metrics.keys():
            self.all_metrics[category].add(key, value)
        else:
            raise

    def get_evaluate_fn(self, key: str):
        if key in self.all_metrics.keys() and isinstance(self.all_metrics[key], EvalMetrics):
            return self.all_metrics[key].evaluate
        else:
            raise

    def get_metric(self, category: str, key: str):
        return self.all_metrics[category][key]

    def report(self):
        return aggregate_unnamed_reports([metrics.report() for metrics in self.all_metrics.values()])

    def reset_metrics(self):
        for metrics in self.all_metrics.values():
            metrics.clear()