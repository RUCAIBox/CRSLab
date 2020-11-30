from crslab.evaluator.metrics import AverageMetric


class RecallMetric(AverageMetric):
    @staticmethod
    def compute(scores, label, k) -> 'RecallMetric':
        return RecallMetric(int(label in scores[:k]))
