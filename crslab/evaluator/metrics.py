# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/24
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

import functools
import math
import re
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from typing import Any, Union, List, Optional, Dict

import torch
from nltk.translate.bleu_score import sentence_bleu
from nltk.util import ngrams

TScalar = Union[int, float, torch.Tensor]
TVector = Union[List[TScalar], torch.Tensor]

re_art = re.compile(r'\b(a|an|the)\b')
re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')
re_space = re.compile(r'\s+')


@functools.total_ordering
class Metric(ABC):
    """
    Base class for storing metrics.

    Subclasses should define .value(). Examples are provided for each subclass.
    """

    @abstractmethod
    def value(self) -> float:
        """
        Return the value of the metric as a float.
        """
        pass

    @abstractmethod
    def __add__(self, other: Any) -> 'Metric':
        raise NotImplementedError

    def __iadd__(self, other):
        return self.__radd__(other)

    def __radd__(self, other: Any):
        if other is None:
            return self
        return self.__add__(other)

    def __str__(self) -> str:
        return f'{self.value():.4g}'

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.value():.4g})'

    def __float__(self) -> float:
        return float(self.value())

    def __int__(self) -> int:
        return int(self.value())

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Metric):
            return self.value() == other.value()
        else:
            return self.value() == other

    def __lt__(self, other: Any) -> bool:
        if isinstance(other, Metric):
            return self.value() < other.value()
        else:
            return self.value() < other

    def __sub__(self, other: Any) -> float:
        """
        Used heavily for assertAlmostEqual.
        """
        if not isinstance(other, float):
            raise TypeError('Metrics.__sub__ is intentionally limited to floats.')
        return self.value() - other

    def __rsub__(self, other: Any) -> float:
        """
        Used heavily for assertAlmostEqual.

        NOTE: This is not necessary in python 3.7+.
        """
        if not isinstance(other, float):
            raise TypeError('Metrics.__rsub__ is intentionally limited to floats.')
        return other - self.value()

    @classmethod
    def as_number(cls, obj: TScalar) -> Union[int, float]:
        if isinstance(obj, torch.Tensor):
            obj_as_number: Union[int, float] = obj.item()
        else:
            obj_as_number = obj  # type: ignore
        assert isinstance(obj_as_number, int) or isinstance(obj_as_number, float)
        return obj_as_number

    @classmethod
    def as_float(cls, obj: TScalar) -> float:
        return float(cls.as_number(obj))

    @classmethod
    def as_int(cls, obj: TScalar) -> int:
        return int(cls.as_number(obj))

    @classmethod
    def many(cls, *objs: List[TVector]) -> List['Metric']:
        """
        Construct many of a Metric from the base parts.

        Useful if you separately compute numerators and denomenators, etc.
        """
        lengths = [len(o) for o in objs]
        if len(set(lengths)) != 1:
            raise IndexError(f'Uneven {cls.__name__} constructions: {lengths}')
        return [cls(*items) for items in zip(*objs)]


class SumMetric(Metric):
    """
    Class that keeps a running sum of some metric.

    Examples of SumMetric include things like "exs", the number of examples seen since
    the last report, which depends exactly on a teacher.
    """

    __slots__ = ('_sum',)

    def __init__(self, sum_: TScalar = 0):
        if isinstance(sum_, torch.Tensor):
            self._sum = sum_.item()
        else:
            assert isinstance(sum_, (int, float))
            self._sum = sum_

    def __add__(self, other: Optional['SumMetric']) -> 'SumMetric':
        # NOTE: hinting can be cleaned up with "from __future__ import annotations" when
        # we drop Python 3.6
        if other is None:
            return self
        full_sum = self._sum + other._sum
        # always keep the same return type
        return type(self)(sum_=full_sum)

    def value(self) -> float:
        return self._sum


class AverageMetric(Metric):
    """
    Class that keeps a running average of some metric.

    Examples of AverageMetrics include hits@1, F1, accuracy, etc. These metrics all have
    per-example values that can be directly mapped back to a teacher.
    """

    __slots__ = ('_numer', '_denom')

    def __init__(self, numer: TScalar, denom: TScalar = 1):
        self._numer = self.as_number(numer)
        self._denom = self.as_number(denom)

    def __add__(self, other: Optional['AverageMetric']) -> 'AverageMetric':
        # NOTE: hinting can be cleaned up with "from __future__ import annotations" when
        # we drop Python 3.6
        if other is None:
            return self
        full_numer: TScalar = self._numer + other._numer
        full_denom: TScalar = self._denom + other._denom
        # always keep the same return type
        return type(self)(numer=full_numer, denom=full_denom)

    def value(self) -> float:
        if self._numer == 0 and self._denom == 0:
            # don't nan out if we haven't counted anything
            return 0.0
        if self._denom == 0:
            return float('nan')
        return self._numer / self._denom


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


class RecallMetric(AverageMetric):
    @staticmethod
    def compute(scores, label, k) -> 'RecallMetric':
        return RecallMetric(int(label in scores[:k]))


def aggregate_unnamed_reports(reports: List[Dict[str, Metric]]) -> Dict[str, Metric]:
    """
    Combines metrics without regard for tracking provenence.
    """
    m: Dict[str, Metric] = {}
    for task_report in reports:
        for each_metric, value in task_report.items():
            m[each_metric] = m.get(each_metric) + value
    return m


class Metrics(object):
    """
    Metrics aggregator.
    """

    def __init__(self):
        self._data = {}

    def __str__(self):
        return str(self._data)

    def __repr__(self):
        return f'Metrics({repr(self._data)})'

    def get(self, key: str):
        if key in self._data.keys():
            return self._data[key].value()
        else:
            raise

    def __getitem__(self, item):
        return self.get(item)

    def add(self, key: str, value: Optional[Metric]) -> None:
        """
        Record an accumulation to a metric.
        """
        self._data[key] = self._data.get(key) + value

    def report(self):
        """
        Report the metrics over all data seen so far.
        """
        return {k: v for k, v in self._data.items()}

    def clear(self):
        """
        Clear all the metrics.
        """
        self._data.clear()


class EvalMetrics(Metrics, ABC):
    @abstractmethod
    def evaluate(self, *args, **kwargs):
        pass


class GenMetrics(EvalMetrics):
    def __init__(self):
        # bleu@1-4, dist@1-4, f1
        super(GenMetrics, self).__init__()
        self.dist_set = defaultdict(set)
        self.dist_cnt = 0

    def evaluate(self, pred: str, labels: List[str]):
        if pred:
            self.add("f1", F1Metric.compute(pred, labels))
            for k in range(1, 5):
                self.add(f"bleu@{k}", BleuMetric.compute(pred, labels, k))
                for token in ngrams(pred, k):
                    self.dist_set[f"dist@{k}"].add(token)
            self.dist_cnt += 1

    def report(self):
        for k, v in self.dist_set.items():
            self.add(k, AverageMetric(len(v) / self.dist_cnt))
        return super(GenMetrics, self).report()

    def clear(self):
        super(GenMetrics, self).clear()
        self.dist_set.clear()
        self.dist_cnt = 0


class RecMetrics(EvalMetrics):
    def __init__(self):
        # recall@1, 10, 50
        super(RecMetrics, self).__init__()

    def evaluate(self, scores: TVector, label: TScalar):
        for k in [1, 10, 50]:
            if len(scores) >= k:
                self.add(f"recall@{k}", RecallMetric.compute(scores, label, k))
