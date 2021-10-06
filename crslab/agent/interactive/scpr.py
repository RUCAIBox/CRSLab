# @Time   : 2021/8/4
# @Author : Chenzhan Shang
# @Email  : czshang@outlook.com

import torch
from tqdm import tqdm

from crslab.agent.interactive.base import InteractiveAgent
from crslab.utils import AgentType


class SCPRAgent(InteractiveAgent):
    agent_type = AgentType.INTERACTIVE

    def __init__(self):
        pass
