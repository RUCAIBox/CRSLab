# @Time   : 2021/10/6
# @Author : Chenzhan Shang
# @Email  : czshang@outlook.com

from enum import Enum


class ModelType(Enum):
    """Type of models

    - ``GENERATION``: Generation-based models.
    - ``RETRIEVAL``: Retrieval-based models.
    """

    GENERATION = 1
    RETRIEVAL = 2


class AgentType(Enum):
    """Type of agents

    - ``SUPERVISED``: Supervised agent for generation-based models.
    - ``INTERACTIVE``: Interactive agent for retrieval-based models.
    """

    SUPERVISED = 1
    INTERACTIVE = 2


class DatasetType(Enum):
    """Type of datasets

    - ``TEXT``: Text-based dataset for supervised agent.
    - ``ATTRIBUTE``: Attribute-based dataset for interactive agent.
    """

    TEXT = 1
    ATTRIBUTE = 2
