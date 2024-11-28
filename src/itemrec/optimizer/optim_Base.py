# -------------------------------------------------------------------
# ItemRec / Item Recommendation Benchmark
# Copyright (C) 2024 Tiny Snow / Weiqin Yang @ Zhejiang University
# -------------------------------------------------------------------
# Module: Model - Base Optimizer
# Description:
#  This module provides the Base Optimizer for ItemRec.
#  All optimizers should be inherited from IROptimizer, the standard
#  and base optimizer class for ItemRec.
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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from ..dataset import IRDataBatch, IRDataset, IRDataLoader
from ..model import IRModel

# public functions --------------------------------------------------
__all__ = [
    'IROptimizer',
]

# IROptimizer -------------------------------------------------------
class IROptimizer(ABC):
    r"""
    ## Class
    The standard and base optimizer class for ItemRec.
    IROptimizer is a wrapper of torch.optim.Optimizer, which mainly uses 
    torch.optim.Adam as the default optimizer. You can also customize
    your own optimizer by inheriting from this class.
    
    ## Methods
    You should inherit from this class to implement your own optimizer, 
    and at least implement the following methods:
    - __init__:
        The constructor of the optimizer.
    - cal_loss:
        Calculate the loss for batch data.
        
    We then provide the following methods:
    - step:
        The step function of the optimizer.
    - zero_grad:
        Zero the gradients of the optimizer.
    """
    optimizer: Optimizer = NotImplemented

    @abstractmethod
    def __init__(self) -> None:
        r"""
        ## Function
        The constructor of the optimizer.
        """
        return NotImplemented
    
    def step(self, batch: IRDataBatch, epoch: int) -> float:
        r"""
        ## Function
        Perform a single optimization step for batch data.

        ## Arguments
        batch: IRDataBatch
            the batch data
        epoch: int
            the current epoch (from 0 to epoch_num - 1)

        ## Returns
        The loss of the batch data.
        """
        self.zero_grad()
        loss = self.cal_loss(batch)
        loss.backward()
        self.optimizer.step()
        return loss.cpu().item()
    
    def zero_grad(self) -> None:
        r"""
        ## Function
        Zero the gradients of the optimizer.
        """
        self.optimizer.zero_grad()

    @abstractmethod
    def cal_loss(self, batch: IRDataBatch, *args, **kwargs) -> torch.Tensor:
        r"""
        ## Function
        Calculate the loss for batch data.

        ## Arguments
        batch: IRDataBatch
            the batch data

        ## Returns
        The loss of the batch data.
        """
        return NotImplemented

