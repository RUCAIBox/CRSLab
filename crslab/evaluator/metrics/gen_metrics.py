import math
import re
from collections import Counter
from typing import List, Optional

from nltk import ngrams
from nltk.translate.bleu_score import sentence_bleu

from crslab.evaluator.metrics.base_metrics import AverageMetric, SumMetric

re_art = re.compile(r'\b(a|an|the)\b')
re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')
re_space = re.compile(r'\s+')


class PPLMetric(AverageMetric):
    def value(self):
        return math.exp(super().value())


def normalize_answer(s):
    """
    Lower text and remove punctuation, articles and extra whitespace.
    """

    s = s.lower()
    s = re_punc.sub(' ', s)
    s = re_art.sub(' ', s)
    s = re_space.sub(' ', s)
    # s = ' '.join(s.split())
    return s


class ExactMatchMetric(AverageMetric):
    @staticmethod
    def compute(guess: str, answers: List[str]) -> 'ExactMatchMetric':
        if guess is None or answers is None:
            return None
        for a in answers:
            if guess == a:
                return ExactMatchMetric(1)
        return ExactMatchMetric(0)


class F1Metric(AverageMetric):
    """
    Helper class which computes token-level F1.
    """

    @staticmethod
    def _prec_recall_f1_score(pred_items, gold_items):
        """
        Compute precision, recall and f1 given a set of gold and prediction items.

        :param pred_items: iterable of predicted values
        :param gold_items: iterable of gold values

        :return: tuple (p, r, f1) for precision, recall, f1
        """
        common = Counter(gold_items) & Counter(pred_items)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(pred_items)
        recall = 1.0 * num_same / len(gold_items)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    @staticmethod
    def compute(guess: str, answers: List[str]) -> 'F1Metric':
        if guess is None or answers is None:
            return AverageMetric(0, 0)
        g_tokens = guess.split()
        scores = [
            F1Metric._prec_recall_f1_score(g_tokens, a.split())
            for a in answers
        ]
        return F1Metric(max(scores), 1)


class BleuMetric(AverageMetric):
    @staticmethod
    def compute(guess: str, answers: List[str], k: int) -> Optional['BleuMetric']:
        """
        Compute approximate BLEU score between guess and a set of answers.
        """

        weights = [0] * 4
        weights[k - 1] = 1
        score = sentence_bleu(
            [a.split(" ") for a in answers],
            guess.split(" "),
            weights=weights,
        )
        return BleuMetric(score)


class DistMetric(SumMetric):
    @staticmethod
    def compute(sent: str, k: int) -> 'DistMetric':
        token_set = set()
        for token in ngrams(sent.split(), k):
            token_set.add(token)
        return DistMetric(len(token_set))
