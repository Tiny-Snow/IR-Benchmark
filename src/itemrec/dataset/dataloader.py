# -------------------------------------------------------------------
# ItemRec / Item Recommendation Benchmark
# Copyright (C) 2024 Tiny Snow / Weiqin Yang @ Zhejiang University
# -------------------------------------------------------------------
# Module: Dataloader
# Description:
#   This module includes Dataloader used in ItemRec.
#   All dataloaders should be inherited from IRDataloader, the standard
#   and base dataloader class for ItemRec.
# -------------------------------------------------------------------

# import modules ----------------------------------------------------
from typing import (
    Any, 
    Optional,
    List,
    Tuple,
    Set,
    Dict,
    Callable,
)
from ..utils import logger
from .dataset import IRDataset
import torch
from torch.utils.data import DataLoader

# public functions --------------------------------------------------
__all__ = [
    'IRData',
    'IRDataBatch',
    'IRDataLoader',
]

# IRData ------------------------------------------------------------
class IRData:
    r"""
    ## Class
    The standard and base data class used for ItemRec training.

    We provide the standard data class for ItemRec, including the
    following attributes:
    - user: torch.Tensor((1), dtype=torch.long)
        the user id
    - pos_item: torch.Tensor((1), dtype=torch.long)
        the positive item id
    - neg_items: torch.Tensor((N), dtype=torch.long)
        the negative item ids, `N` is the number of negative items

    Note that `IRData` is used for training, and the test process
    is in fact a ranking process, i.e. for each user, we infer the
    scores of all items and rank them to get the Top-K items. Thus,
    you should not use `IRData` for testing.
    """
    def __init__(self, user: torch.Tensor, pos_item: torch.Tensor, neg_items: torch.Tensor):
        r"""
        ## Function
        Initialize the IRData object.

        ## Arguments
        user: torch.Tensor((1), dtype=torch.long)
            the user id
        pos_item: torch.Tensor((1), dtype=torch.long)
            the positive item id
        neg_items: torch.Tensor((N), dtype=torch.long)
            the negative item ids, `N` is the number of negative items
        """
        self.user = user
        self.pos_item = pos_item
        self.neg_items = neg_items
    
    def to(self, device: torch.device):
        self.user = self.user.to(device)
        self.pos_item = self.pos_item.to(device)
        self.neg_items = self.neg_items.to(device)

    def __str__(self):
        return f'IRData(user={self.user.shape}, pos_item={self.pos_item.shape}, neg_items={self.neg_items.shape})'

class IRDataBatch:
    r"""
    ## Class
    The standard and base data batch class used for ItemRec training.

    We provide the standard data batch class for ItemRec, including the
    following attributes:
    - user: torch.Tensor((B), dtype=torch.long)
        the user ids
    - pos_item: torch.Tensor((B), dtype=torch.long)
        the positive item ids
    - neg_items: torch.Tensor((B, N), dtype=torch.long)
        the negative item ids, `N` is the number of negative items

    Note that `IRDataBatch` is only used for training.
    """
    def __init__(self, user: torch.Tensor, pos_item: torch.Tensor, neg_items: torch.Tensor):
        r"""
        ## Function
        Initialize the IRDataBatch object.

        ## Arguments
        user: torch.Tensor((B), dtype=torch.long)
            the user ids
        pos_item: torch.Tensor((B), dtype=torch.long)
            the positive item ids
        neg_items: torch.Tensor((B, N), dtype=torch.long)
            the negative item ids, `N` is the number of negative items
        """
        self.user = user
        self.pos_item = pos_item
        self.neg_items = neg_items
    
    def __len__(self):
        return len(self.user)
    
    def to(self, device: torch.device):
        self.user = self.user.to(device)
        self.pos_item = self.pos_item.to(device)
        self.neg_items = self.neg_items.to(device)

    def __str__(self):
        return f'IRDataBatch(user={self.user.shape}, pos_item={self.pos_item.shape}, neg_items={self.neg_items.shape})'


# Other IRData Classes ----------------------------------------------



# IRDataloader -------------------------------------------------------
class IRDataLoader(DataLoader):
    r"""
    ## Class
    The standard and base dataloader class for ItemRec training.

    We provide the standard dataloader for ItemRec, which is inherited 
    from `torch.utils.data.DataLoader`. However, the `IRDataLoader` is
    only used for training.

    In training, one data sample is a `IRData` object, including the
    `(user, pos_item)` pair we selected from the `IRDataset.train_interactions`
    with the negative items `neg_items` we sampled randomly in the runtime.
    Thus, we implement the `collate_fn` to generate batch data `IRDataBatch`.
    """
    def __init__(self, dataset: IRDataset, batch_size: int = 1, shuffle: bool = False, 
        num_workers: int = 0, drop_last: bool = False, neg_num: int = 0):
        r"""
        ## Function
        Initialize the IRDataloader.

        ## Arguments
        dataset: IRDataset
            the IRDataset object you processed correctly.
        batch_size: int (default: 1)
            the batch size.
        shuffle: bool (default: False)
            whether to shuffle the data.
        num_workers: int (default: 0)
            the number of workers.
        drop_last: bool (default: False)
            whether to drop the last incomplete batch.
        neg_num: int (default: 0)
            the number of negative items for each user in the batch.
        """
        super(IRDataLoader, self).__init__(
            dataset = dataset.train_interactions,
            batch_size = batch_size,
            shuffle = shuffle,
            num_workers = num_workers,
            collate_fn = lambda x: self._collate_fn(x),
            drop_last = drop_last
        )
        self._dataset = dataset # not same as `dataset` in `DataLoader`
        self.neg_num = neg_num

    def _collate_fn(self, batch: List[Tuple[int, int]]) -> IRDataBatch:
        r"""
        ## Function
        The collate function to generate batch data `IRDataBatch`.

        ## Arguments
        batch: List[Tuple[int, int]]
            the batch data, each element is a tuple `(user, pos_item)`.

        ## Returns
        IRDataBatch
            the batch data `IRDataBatch`.
        """
        user = torch.tensor([x[0] for x in batch], dtype=torch.long)
        pos_item = torch.tensor([x[1] for x in batch], dtype=torch.long)
        neg_items = torch.tensor(
            [self._dataset.sample_negative(u, self.neg_num) for u in user],
            dtype=torch.long
        )
        return IRDataBatch(user, pos_item, neg_items)

