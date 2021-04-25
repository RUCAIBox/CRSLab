# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/24, 2020/12/2
# @Author : Kun Zhou, Xiaolei Wang
# @Email  : francis_kun_zhou@163.com, wxl1999@foxmail.com

import functools
from abc import ABC, abstractmethod

import torch
from typing import Any, Union, List, Optional, Dict

TScalar = Union[int, float, torch.Tensor]
TVector = Union[List[TScalar], torch.Tensor]


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
