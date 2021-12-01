# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/24, 2020/12/18
# @Author : Kun Zhou, Xiaolei Wang
# @Email  : francis_kun_zhou@163.com, wxl1999@foxmail.com

# UPDATE:
# @Time   : 2021/10/05
# @Author  :   Zhipeng Zhao
# @email   :   oran_official@outlook.com

import torch


def compute_grad_norm(parameters, norm_type=2.0):
    """
    Compute norm over gradients of model parameters.

    :param parameters:
        the model parameters for gradient norm calculation. Iterable of
        Tensors or single Tensor
    :param norm_type:
        type of p-norm to use

    :returns:
        the computed gradient norm
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p is not None and p.grad is not None]
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    return total_norm ** (1.0 / norm_type)


def ind2txt(inds, ind2tok, end_token_idx=None, unk_token='unk'):
    sentence = []
    for ind in inds:
        if isinstance(ind, torch.Tensor):
            ind = ind.item()
        if end_token_idx and ind == end_token_idx:
            break
        sentence.append(ind2tok.get(ind, unk_token))
    return ' '.join(sentence)

def ind2txt_with_slots(inds,slots,ind2tok, end_token_idx=None, unk_token='unk',slot_token='[ITEM]'):
    sentence = []
    for ind in inds:
        if isinstance(ind, torch.Tensor):
            ind = ind.item()
        if end_token_idx and ind == end_token_idx:
            break
        token = ind2tok.get(ind, unk_token)
        if token == slot_token:
            token = slots[0]
            slots = slots[1:] 
        sentence.append(token)
    return ' '.join(sentence)

def ind2slot(inds,ind2slot):
    return [ ind2slot[ind] for ind in inds]
