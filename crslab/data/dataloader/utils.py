# -*- encoding: utf-8 -*-
# @Time    :   2020/12/10
# @Author  :   Xiaolei Wang
# @email   :   wxl1999@foxmail.com

# UPDATE
# @Time    :   2020/12/20, 2020/12/15
# @Author  :   Xiaolei Wang, Yuanhang Zhou
# @email   :   wxl1999@foxmail.com, sdzyh002@gmail

# UPDATE
# @Time   : 2021/10/06
# @Author : Zhipeng Zhao
# @Email  : oran_official@outlook.com


from copy import copy

import torch
from typing import List, Union, Optional


def padded_tensor(
        items: List[Union[List[int], torch.LongTensor]],
        pad_idx: int = 0,
        pad_tail: bool = True,
        max_len: Optional[int] = None,
) -> torch.LongTensor:
    """Create a padded matrix from an uneven list of lists.

    Returns padded matrix.

    Matrix is right-padded (filled to the right) by default, but can be
    left padded if the flag is set to True.

    Matrix can also be placed on cuda automatically.

    :param list[iter[int]] items: List of items
    :param int pad_idx: the value to use for padding
    :param bool pad_tail:
    :param int max_len: if None, the max length is the maximum item length

    :returns: padded tensor.
    :rtype: Tensor[int64]

    """
    # number of items
    n = len(items)
    # length of each item
    lens: List[int] = [len(item) for item in items]  # type: ignore
    # max in time dimension
    t = max(lens) if max_len is None else max_len
    # if input tensors are empty, we should expand to nulls
    t = max(t, 1)

    if isinstance(items[0], torch.Tensor):
        # keep type of input tensors, they may already be cuda ones
        output = items[0].new(n, t)  # type: ignore
    else:
        output = torch.LongTensor(n, t)  # type: ignore
    output.fill_(pad_idx)

    for i, (item, length) in enumerate(zip(items, lens)):
        if length == 0:
            # skip empty items
            continue
        if not isinstance(item, torch.Tensor):
            # put non-tensors into a tensor
            item = torch.tensor(item, dtype=torch.long)  # type: ignore
        if pad_tail:
            # place at beginning
            output[i, :length] = item
        else:
            # place at end
            output[i, t - length:] = item

    return output


def get_onehot(data_list, categories) -> torch.Tensor:
    """Transform lists of label into one-hot.

    Args:
        data_list (list of list of int): source data.
        categories (int): #label class.

    Returns:
        torch.Tensor: one-hot labels.

    """
    onehot_labels = []
    for label_list in data_list:
        onehot_label = torch.zeros(categories)
        for label in label_list:
            onehot_label[label] = 1.0 / len(label_list)
        onehot_labels.append(onehot_label)
    return torch.stack(onehot_labels, dim=0)


def add_start_end_token_idx(vec: list, start_token_idx: int = None, end_token_idx: int = None):
    """Can choose to add start token in the beginning and end token in the end.

    Args:
        vec: source list composed of indexes.
        start_token_idx: index of start token.
        end_token_idx: index of end token.

    Returns:
        list: list added start or end token index.

    """
    res = copy(vec)
    if start_token_idx:
        res.insert(0, start_token_idx)
    if end_token_idx:
        res.append(end_token_idx)
    return res


def truncate(vec, max_length, truncate_tail=True):
    """truncate vec to make its length no more than max length.

    Args:
        vec (list): source list.
        max_length (int)
        truncate_tail (bool, optional): Defaults to True.

    Returns:
        list: truncated vec.

    """
    if max_length is None:
        return vec
    if len(vec) <= max_length:
        return vec
    if max_length == 0:
        return []
    if truncate_tail:
        return vec[:max_length]
    else:
        return vec[-max_length:]


def merge_utt(conversation, split_token_idx=None, keep_split_in_tail=False, final_token_idx=None):
    """merge utterances in one conversation.

    Args:
        conversation (list of list of int): conversation consist of utterances consist of tokens.
        split_token_idx (int): index of split token. Defaults to None.
        keep_split_in_tail (bool): split in tail or head. Defaults to False.
        final_token_idx (int): index of final token. Defaults to None.

    Returns:
        list: tokens of all utterances in one list.

    """
    merged_conv = []
    for utt in conversation:
        for token in utt:
            merged_conv.append(token)
        if split_token_idx:
            merged_conv.append(split_token_idx)
    if split_token_idx and not keep_split_in_tail:
        merged_conv = merged_conv[:-1]
    if final_token_idx:
        merged_conv.append(final_token_idx)
    return merged_conv

def merge_utt_replace(conversation,detect_token=None,replace_token=None,method="in"):
    if method == 'in': 
        replaced_conv = []
        for utt in conversation:
            for token in utt:
                if detect_token in token:
                    replaced_conv.append(replace_token)
                else:
                    replaced_conv.append(token)
        return replaced_conv
    else:
        return [token.replace(detect_token,replace_token) for utt in conversation for token in utt]
