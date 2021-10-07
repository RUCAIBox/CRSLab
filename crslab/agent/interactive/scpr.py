# @Time   : 2021/8/4
# @Author : Chenzhan Shang
# @Email  : czshang@outlook.com

import torch
from tqdm import tqdm

from crslab.agent.interactive.base import InteractiveAgent
from crslab.utils import AgentType


class SCPRAgent(InteractiveAgent):
    def __init__(self):
        pass

    def _set_agent_type(self) -> AgentType:
        return AgentType.INTERACTIVE
