# -------------------------------------------------------------------
# ItemRec / Item Recommendation Benchmark
# Copyright (C) 2024 Tiny Snow / Weiqin Yang @ Zhejiang University
# -------------------------------------------------------------------
# Module: Sampler
# Description:
#   This module includes the standard Sampler for Dataset used in 
#   ItemRec. To enhance the performance of the training process, we
#   provide the C++ version of the Sampler in ItemRec.
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
from abc import ABC, abstractmethod
import os
import csv
import random
import torch
import torch.utils.cpp_extension as torch_cpp

# public functions --------------------------------------------------
__all__ = [
    'IRSampler',
    'UniformSampler',
]

# IR Sampler --------------------------------------------------------
class IRSampler(ABC):
    r"""
    ## Class
    The standard and base sampler class for ItemRec.
    IRSampler is used for negative sampling in IRDataset.
    
    In general, in every sampling process, IRSampler receives a 
    specific cumulative distribution of items for each user, and 
    samples N negative items based on this distribution.
    """
    def __init__(self) -> None:
        r"""
        ## Function
        Initialize the IRSampler object.
        """
        pass

    @abstractmethod
    def sample(self, user_ids: torch.Tensor, N: int) -> torch.Tensor:
        r"""
        ## Function
        Sample negative items.
        
        ## Arguments
        - user_ids: torch.Tensor((B), dtype=torch.long)
            the user ids
        - N: int
            the number of negative items to sample
        
        ## Returns
        - neg_items: torch.Tensor((B, N), dtype=torch.long)
            the negative item ids, `B` is the number of users, `N` is the number of negative items
        """
        pass
    
# Uniform Sampler ---------------------------------------------------
class UniformSampler(IRSampler):
    r"""
    ## Class
    The standard uniform sampler for ItemRec.
    UniformSampler samples negative items uniformly.
    """
    def __init__(self, user_size: int, item_size: int, pos_items: List[List[int]]) -> None:
        r"""
        ## Function
        Initialize the UniformSampler object.
        
        ## Arguments
        - user_size: int
            the number of users.
        - item_size: int
            the number of items
        - pos_items: List[List[int]]
            the positive item ids for each user, which should be masked during sampling
        """
        super(UniformSampler, self).__init__()
        self.sampler = torch_cpp.load(name="uniform_sample", sources=["itemrec/dataset/sampler_cpp/uniform_sampler.cpp"]).uniform_sample
        self.user_size = user_size
        self.item_size = item_size
        self.pos_items = [set(items) for items in pos_items]
        
    def sample(self, user_ids: torch.Tensor, N: int) -> torch.Tensor:
        r"""
        ## Function
        Sample negative items.
        
        ## Arguments
        - user_ids: torch.Tensor((B), dtype=torch.long)
            the user ids
        - N: int
            the number of negative items to sample
        
        ## Returns
        - neg_items: torch.Tensor((B, N), dtype=torch.long)
            the negative item ids, `B` is the number of users, `N` is the number of negative items
        """
        pos_items = [self.pos_items[user_id] for user_id in user_ids]
        return self.sampler(self.item_size, pos_items, N, random.randint(0, 1000000))

