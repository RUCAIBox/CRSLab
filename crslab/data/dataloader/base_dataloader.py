# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/23, 2020/11/27
# @Author : Kun Zhou, Xiaolei Wang
# @Email  : francis_kun_zhou@163.com, wxl1999@foxmail.com

import random
from abc import ABC
from math import ceil
from typing import List, Tuple, Optional, Union

import torch


class BaseDataLoader(ABC):
    """:class:`BaseDataLoader` is an base object which would return a batch of data which is loaded by
        :class:`~recbole.data.interaction.Interaction` when it is iterated.
        And it is also the ancestor of all other dataloader.

        Args:
            config (Config): The config of dataloader.
            dataset (Dataset): The dataset of dataloader.
            batch_size (int, optional): The batch_size of dataloader. Defaults to ``1``.
            dl_format (InputType, optional): The input type of dataloader. Defaults to
                :obj:`~recbole.utils.enum_type.InputType.POINTWISE`.
            shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.

        Attributes:
            dataset (Dataset): The dataset of this dataloader.
            shuffle (bool): If ``True``, dataloader will shuffle before every epoch.
            real_time (bool): If ``True``, dataloader will do data pre-processing,
                such as neg-sampling and data-augmentation.
            pr (int): Pointer of dataloader.
            step (int): The increment of :attr:`pr` for each batch.
            batch_size (int): The max interaction number for all batch.
        """

    def __init__(self, config, dataset):
        self.config = config
        self.dataset = dataset

    @staticmethod
    def batch_split(dataset, batch_size, shuffle=False):
        batch_num = ceil(len(dataset) / batch_size)
        idx_list = list(range(len(dataset)))
        if shuffle:
            random.shuffle(idx_list)
        for start_idx in range(batch_num):
            batch_idx = idx_list[start_idx * batch_size: (start_idx + 1) * batch_size]
            yield [dataset[idx] for idx in batch_idx]

    def get_data(self, batch_fn, batch_size, shuffle=False, process_fn=None):
        dataset = self.dataset
        if process_fn is not None:
            dataset = process_fn(dataset)
        batches_iteration = self.batch_split(dataset, batch_size, shuffle)
        return [batch_fn(batch) for batch in batches_iteration]

    def get_conv_data(self, batch_size, shuffle=False):
        return self.get_data(self.conv_batchify, batch_size, shuffle, self.conv_process_fn)

    def get_rec_data(self, batch_size, shuffle=False):
        return self.get_data(self.rec_batchify, batch_size, shuffle, self.rec_process_fn)

    def get_guide_data(self, batch_size, shuffle=False):
        return self.get_data(self.guide_batchify, batch_size, shuffle, self.guide_process_fn)

    def conv_process_fn(self, *args, **kwargs):
        return self.dataset

    def conv_batchify(self, *args, **kwargs):
        raise NotImplementedError('dataloader must implement conv_batchify() method')

    def rec_process_fn(self, *args, **kwargs):
        return self.dataset

    def rec_batchify(self, *args, **kwargs):
        raise NotImplementedError('dataloader must implement rec_batchify() method')

    def guide_process_fn(self, *args, **kwargs):
        return self.dataset

    def guide_batchify(self, *args, **kwargs):
        raise NotImplementedError('dataloader must implement guide_batchify() method')

    @staticmethod
    def get_side_data(data, type='RGCN'):
        if type == 'RGCN':
            edge_sets = torch.tensor(data, dtype=torch.long)
            edge_idx = edge_sets[:, :2].t()
            edge_type = edge_sets[:, 2]
            return edge_idx, edge_type
        elif type == 'GCN':
            edge_set = [[co[0] for co in data], [co[1] for co in data]]
            return torch.tensor(edge_set, dtype=torch.long)
        else:
            raise NotImplementedError('type {} has not been implemented', type)

    @staticmethod
    def padded_tensor(
            items: List[Union[List[int], torch.LongTensor]],
            pad_idx: int = 0,
            right_padded: bool = True,
            max_len: Optional[int] = None,
    ) -> Tuple[torch.LongTensor, List[int]]:
        """
        Create a padded matrix from an uneven list of lists.

        Returns (padded, lengths), where padded is the padded matrix, and lengths
        is a list containing the lengths of each row.

        Matrix is right-padded (filled to the right) by default, but can be
        left padded if the flag is set to True.

        Matrix can also be placed on cuda automatically.

        :param list[iter[int]] items: List of items
        :param int pad_idx: the value to use for padding
        :param bool right_padded:
        :param int max_len: if None, the max length is the maximum item length

        :returns: (padded, lengths) tuple
        :rtype: (Tensor[int64], list[int])
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
            if right_padded:
                # place at beginning
                output[i, :length] = item
            else:
                # place at end
                output[i, t - length:] = item

        return output, lens

    @staticmethod
    def get_onehot_label(label_lists, categories):
        onehot_labels = []
        for label_list in label_lists:
            onehot_label = torch.zeros(categories)
            for label in label_list:
                onehot_label[label] = 1.0 / len(label_list)
            onehot_labels.append(onehot_label)
        return torch.stack(onehot_labels, dim=0)
