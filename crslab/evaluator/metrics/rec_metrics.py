from crslab.evaluator.metrics.base_metrics import AverageMetric


class RecallMetric(AverageMetric):
    @staticmethod
    def compute(scores, label, k) -> 'RecallMetric':
        return RecallMetric(int(label in scores[:k]))
