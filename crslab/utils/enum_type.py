# @Time   : 2021/10/6
# @Author : Chenzhan Shang
# @Email  : czshang@outlook.com

from enum import Enum


class ModelType(Enum):
    """Type of models

    - ``GENERATION``: Generation-based models and supervised agent.
    - ``RETRIEVAL``: Retrieval-based models and interactive agent.
    """

    GENERATION = 1
    RETRIEVAL = 2
