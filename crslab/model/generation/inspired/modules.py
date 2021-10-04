# @Time   : 2021/3/10
# @Author : Beichen Zhang
# @Email  : zhangbeichen724@gmail.com

import torch
import torch.nn as nn
import torch.nn.functional as F


class SequenceCrossEntropyLoss(nn.Module):
    """

    Attributes:
        ignore_index: indices corresponding tokens which should be ignored in calculating loss.
        label_smoothing: determine smoothing value in cross entropy loss. should be less than 1.0.

    """

    def __init__(self, ignore_index=None, label_smoothing=-1):
        super().__init__()
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(self, logits, labels):
        """

        Args:
            logits: (batch_size, max_seq_len, vocal_size)
            labels: (batch_size, max_seq_len)

        """
        if self.label_smoothing > 1.0:
            raise ValueError('The param label_smoothing should be in the range of 0.0 to 1.0.')
        if self.ignore_index == None:
            mask = torch.ones_like(labels, dtype=torch.float)
        else:
            mask = (labels != self.ignore_index).float()
        logits_flat = logits.reshape(-1, logits.size(-1))  # (b_s * s_l, num_classes)
        log_probs_flat = F.log_softmax(logits_flat, dim=-1)
        labels_flat = labels.reshape(-1, 1).long()  # (b_s * s_l, 1)

        if self.label_smoothing > 0.0:
            num_classes = logits.size(-1)
            smoothing_value = self.label_smoothing / float(num_classes)
            one_hot_labels = torch.zeros_like(log_probs_flat).scatter_(-1, labels_flat,
                                                                       1.0 - self.label_smoothing)  # fill all the correct indices with 1 - smoothing value.
            smoothed_labels = one_hot_labels + smoothing_value
            negative_log_likelihood_flat = -log_probs_flat * smoothed_labels
            negative_log_likelihood_flat = negative_log_likelihood_flat.sum(-1, keepdim=True)
        else:
            negative_log_likelihood_flat = -torch.gather(log_probs_flat, dim=1, index=labels_flat)  # (b_s * s_l, 1)

        negative_log_likelihood = negative_log_likelihood_flat.view(-1, logits.shape[1])  # (b_s, s_l)
        loss = negative_log_likelihood * mask

        loss = loss.sum(1) / (mask.sum(1) + 1e-13)
        loss = loss.mean()

        return loss
