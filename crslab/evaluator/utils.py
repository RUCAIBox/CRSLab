# -*- encoding: utf-8 -*-
# @Time    :   2020/12/17
# @Author  :   Xiaolei Wang
# @email   :   wxl1999@foxmail.com

# UPDATE
# @Time    :   2020/12/17
# @Author  :   Xiaolei Wang
# @email   :   wxl1999@foxmail.com

import json
import re
import shutil
from collections import OrderedDict

import math
import torch
from typing import Union, Tuple

from .metrics import Metric


def _line_width():
    try:
        # if we're in an interactive ipython notebook, hardcode a longer width
        __IPYTHON__
        return 128
    except NameError:
        return shutil.get_terminal_size((88, 24)).columns


def float_formatter(f: Union[float, int]) -> str:
    """
    Format a float as a pretty string.
    """
    if f != f:
        # instead of returning nan, return "" so it shows blank in table
        return ""
    if isinstance(f, int):
        # don't do any rounding of integers, leave them alone
        return str(f)
    if f >= 1000:
        # numbers > 1000 just round to the nearest integer
        s = f'{f:.0f}'
    else:
        # otherwise show 4 significant figures, regardless of decimal spot
        s = f'{f:.4g}'
    # replace leading 0's with blanks for easier reading
    # example:  -0.32 to -.32
    s = s.replace('-0.', '-.')
    if s.startswith('0.'):
        s = s[1:]
    # Add the trailing 0's to always show 4 digits
    # example: .32 to .3200
    if s[0] == '.' and len(s) < 5:
        s += '0' * (5 - len(s))
    return s


def round_sigfigs(x: Union[float, 'torch.Tensor'], sigfigs=4) -> float:
    """
    Round value to specified significant figures.

    :param x: input number
    :param sigfigs: number of significant figures to return

    :returns: float number rounded to specified sigfigs
    """
    x_: float
    if isinstance(x, torch.Tensor):
        x_ = x.item()
    else:
        x_ = x  # type: ignore

    try:
        if x_ == 0:
            return 0
        return round(x_, -math.floor(math.log10(abs(x_)) - sigfigs + 1))
    except (ValueError, OverflowError) as ex:
        if x_ in [float('inf'), float('-inf')] or x_ != x_:  # inf or nan
            return x_
        else:
            raise ex


def _report_sort_key(report_key: str) -> Tuple[str, str]:
    """
    Sorting name for reports.

    Sorts by main metric alphabetically, then by task.
    """
    # if metric is on its own, like "f1", we will return ('', 'f1')
    # if metric is from multitask, we denote it.
    # e.g. "convai2/f1" -> ('convai2', 'f1')
    # we handle multiple cases of / because sometimes teacher IDs have
    # filenames.
    fields = report_key.split("/")
    main_key = fields.pop(-1)
    sub_key = '/'.join(fields)
    return (sub_key or 'all', main_key)


def nice_report(report) -> str:
    """
    Render an agent Report as a beautiful string.

    If pandas is installed,  we will use it to render as a table. Multitask
    metrics will be shown per row, e.g.

    .. code-block:
                 f1   ppl
       all     .410  27.0
       task1   .400  32.0
       task2   .420  22.0

    If pandas is not available, we will use a dict with like-metrics placed
    next to each other.
    """
    if not report:
        return ""

    try:
        import pandas as pd

        use_pandas = True
    except ImportError:
        use_pandas = False

    sorted_keys = sorted(report.keys(), key=_report_sort_key)
    output: OrderedDict[Union[str, Tuple[str, str]], float] = OrderedDict()
    for k in sorted_keys:
        v = report[k]
        if isinstance(v, Metric):
            v = v.value()
        if use_pandas:
            output[_report_sort_key(k)] = v
        else:
            output[k] = v

    if use_pandas:
        line_width = _line_width()

        df = pd.DataFrame([output])
        df.columns = pd.MultiIndex.from_tuples(df.columns)
        df = df.stack().transpose().droplevel(0, axis=1)
        result = "   " + df.to_string(
            na_rep="",
            line_width=line_width - 3,  # -3 for the extra spaces we add
            float_format=float_formatter,
            index=df.shape[0] > 1,
        ).replace("\n\n", "\n").replace("\n", "\n   ")
        result = re.sub(r"\s+$", "", result)
        return result
    else:
        return json.dumps(
            {
                k: round_sigfigs(v, 4) if isinstance(v, float) else v
                for k, v in output.items()
            }
        )
