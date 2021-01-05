# -*- encoding: utf-8 -*-
# @Time    :   2020/11/26
# @Author  :   Xiaolei Wang
# @email   :   wxl1999@foxmail.com

# UPDATE
# @Time    :   2020/11/16
# @Author  :   Xiaolei Wang
# @email   :   wxl1999@foxmail.com

import torch


def edge_to_pyg_format(edge, type='RGCN'):
    if type == 'RGCN':
        edge_sets = torch.as_tensor(edge, dtype=torch.long)
        edge_idx = edge_sets[:, :2].t()
        edge_type = edge_sets[:, 2]
        return edge_idx, edge_type
    elif type == 'GCN':
        edge_set = [[co[0] for co in edge], [co[1] for co in edge]]
        return torch.as_tensor(edge_set, dtype=torch.long)
    else:
        raise NotImplementedError('type {} has not been implemented', type)


def sort_for_packed_sequence(lengths: torch.Tensor):
    """
    :param lengths: 1D array of lengths
    :return: sorted_lengths (lengths in descending order), sorted_idx (indices to sort), rev_idx (indices to retrieve original order)

    """
    sorted_idx = torch.argsort(lengths, descending=True)  # idx to sort by length
    rev_idx = torch.argsort(sorted_idx)  # idx to retrieve original order
    sorted_lengths = lengths[sorted_idx]

    return sorted_lengths, sorted_idx, rev_idx
