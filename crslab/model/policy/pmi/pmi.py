# @Time   : 2020/12/17
# @Author : Yuanhang Zhou
# @Email  : sdzyh002@gmail

# UPDATE
# @Time   : 2020/12/29, 2021/1/4
# @Author : Xiaolei Wang, Yuanhang Zhou
# @email  : wxl1999@foxmail.com, sdzyh002@gmail.com

r"""
PMI
===
"""

from collections import defaultdict

import torch

from crslab.model.base import BaseModel


class PMIModel(BaseModel):
    """

    Attributes:
        topic_class_num: A integer indicating the number of topic.
        pad_topic: A integer indicating the id of topic padding.

    """

    def __init__(self, opt, device, vocab, side_data):
        """

        Args:
            opt (dict): A dictionary record the hyper parameters.
            device (torch.device): A variable indicating which device to place the data and model.
            vocab (dict): A dictionary record the vocabulary information.
            side_data (dict): A dictionary record the side data.

        """
        self.topic_class_num = vocab['n_topic']
        self.pad_topic = vocab['pad_topic']
        super(PMIModel, self).__init__(opt, device)

    def build_model(self, *args, **kwargs):
        """build model"""
        self.topic_to_num = defaultdict(int)
        self.t2gram_to_num = defaultdict(int)
        self.last_topic_to_target_topic = defaultdict(int)

    def forward(self, batch, mode):
        # conv_id, message_id, context, context_mask, topic_path_kw, tp_mask, user_profile, profile_mask, y = batch
        context, context_mask, topic_path_kw, tp_mask, user_profile, profile_mask, target = batch

        if mode == 'train':
            for topic_path in topic_path_kw:
                topic_path = [topic_id.item() for topic_id in topic_path if topic_id.item() != self.pad_topic]
                for topic in topic_path:
                    self.topic_to_num[topic] += 1
                for i in range(1, len(topic_path)):
                    self.t2gram_to_num[(topic_path[i - 1], topic_path[i])] += 1
                self.last_topic_to_target_topic[(topic_path[-1], target[0])] += 1

        test_last_topic_to_target_topic = defaultdict(int)
        for topic_path in topic_path_kw:
            topic_path = [topic_id.item() for topic_id in topic_path if topic_id.item() != self.pad_topic]
            test_last_topic_to_target_topic[(topic_path[-1], target[0])] += 1

        total_1_gram = sum(self.topic_to_num.values())
        total_2_gram = sum(self.t2gram_to_num.values())
        p_1_gram = {topic: num / total_1_gram for topic, num in self.topic_to_num.items()}
        p_2_gram = {topic_tuple: num / total_2_gram for topic_tuple, num in self.t2gram_to_num.items()}

        topic_scores = []
        for (last_topic, target_topic), num in test_last_topic_to_target_topic.items():
            candidate_topic_to_PMI = {}
            for cnad_topic in self.topic_to_num:
                if (last_topic, cnad_topic) in p_2_gram:
                    candidate_topic_to_PMI[cnad_topic] = p_2_gram.get((last_topic, cnad_topic), 0) / (
                            p_1_gram.get(last_topic, 0) * p_1_gram.get(cnad_topic, 0))
            top_cand = dict(sorted(candidate_topic_to_PMI.items(), key=lambda x: x[1], reverse=True))
            # top_cand = [topic for topic, num in top_cand]
            topic_scores.append([top_cand.get(topic_id, 0) for topic_id in range(self.topic_class_num)])

        return None, torch.tensor(topic_scores, dtype=torch.long)
