# @Time   : 2021/8/11
# @Author : Chenzhan Shang
# @Email  : czshang@outlook.com

# define sender type
USER = 'USER'
SYSTEM = 'SYSTEM'

# define system behavior
ASK = 'ASK'
REC = 'REC'
START = 'START'

# define user behavior
INFORM = 'INFORM'
ACCEPT = 'ACCEPT'
REJECT = 'REJECT'


def run_one_episode(user, item, start_attribute, kg, mode):
    """Run one episode and return a tuple according to the current mode.

    Returns:
        if mode == 'train': (user, pos_item, neg_item, attr_neg_item, preferred_attr)
        elif mode == 'valid': (user, pos_item, candidate_items, preferred_attr)

    """
    pass


class Message:
    """The data structure for message."""
    def __init__(self, sender, receiver, message_type, data):
        self.sender = sender
        self.receiver = receiver
        self.message_type = message_type
        self.data = data


class User:
    """User simulator interact with the rule-based system."""
    def __init__(self, user_id, item_id, kg):
        self.user_id = user_id
        self.item_id = item_id  # ground-truth item
        self.candidate_items = list(kg['item'].keys())  # len 7432

    def inform_attribute(self, attribute):
        pass

    def response(self, message):
        pass


class System:
    """Rule-based system interact with user simulator."""
    def __init__(self):
        pass

    def response(self):
        pass
