# -------------------------------------------------------------------
# ItemRec / Item Recommendation Benchmark
# Copyright (C) 2024 Tiny Snow / Weiqin Yang @ Zhejiang University
# -------------------------------------------------------------------
# Module: Model - LambdaLoss Optimizer
# Description:
#  This module provides the LambdaLoss Optimizer for ItemRec. LambdaLoss
#  is a pairwise NDCG optimization algorithm similar to LambdaRank.
#  - Wang, X., Li, C., Golbandi, N., Bendersky, M., & Najork, M. (2018, October). 
#   The lambdaloss framework for ranking metric optimization. 
#   In Proceedings of the 27th ACM international conference on information and knowledge management (pp. 1313-1322).
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
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .optim_Base import IROptimizer
from ..dataset import IRDataBatch
from ..model import IRModel
from .optim_LambdaRank import LambdaRankOptimizer

# public functions --------------------------------------------------
__all__ = [
    'LambdaLossOptimizer',
]

# LambdaLossOptimizer -----------------------------------------------
class LambdaLossOptimizer(LambdaRankOptimizer):
    r"""
    ## Class
    The LambdaLoss Optimizer for ItemRec.
    LambdaLoss is a pairwise NDCG optimization algorithm similar to LambdaRank.
    The LambdaLoss optimizer is inherited from LambdaRankOptimizer.
    
    ## Algorithms
    The only difference between LambdaLoss and LambdaRank is the definition of
    $\Delta(i, j)$, which is defined as follows:

    $$
    \Delta(i, j) = |G(y_i) - G(y_j)||1 / D_{|\pi_i - \pi_j|} - 1 / D_{|\pi_i - \pi_j| + 1}|
    $$

    where $G(y_i) = 2^{y_i} - 1$ is the gain of item i ($y_i = 1$ if $(u, i)$ is a positive pair,
    otherwise $y_i = 0$), and $D(\pi_i) = \log_2(1 + \pi_i)$ is the discount of item i, where $\pi_i$
    is the rank of item i.
    
    NOTE: For every user u and positive-negative pair (i, j) in a batch, we need to calculate
    the ranking of i and j, with the complexity of O(|U||I|). Thus, LambdaLoss is not suitable 
    for large-scale datasets.

    NOTE: Since we only sample one negative item $j_0$ for each positive item, we approximate the
    sum $\sum_{y_i > y_j} \Delta(i, j) \cdot \sigma(d_{uij})$ with $1 / D_{|\pi_i - \pi_{j_0}|} 
    \cdot \sigma(d_{uij})$.

    ## References
    - Wang, X., Li, C., Golbandi, N., Bendersky, M., & Najork, M. (2018, October).
        The lambdaloss framework for ranking metric optimization.
        In Proceedings of the 27th ACM international conference on information and knowledge management (pp. 1313-1322).
    """
    def __init__(self, model: IRModel, lr: float = 0.001, weight_decay: float = 0.0) -> None:
        r"""
        ## Function
        The constructor of the LambdaLoss Optimizer.
        
        ## Arguments
        model: IRModel
            the model to be optimized
        lr: float
            the learning rate
        weight_decay: float
            the weight decay parameter
        """
        super(LambdaLossOptimizer, self).__init__(model, lr, weight_decay)

    def cal_loss(self, batch: IRDataBatch) -> torch.Tensor:
        r"""
        ## Function
        Calculate the LambdaLoss loss for batch data.
        
        ## Arguments
        batch: IRDataBatch
            the batch data, with shapes:
            - user: torch.Tensor((B), dtype=torch.long)
                the user ids
            - pos_item: torch.Tensor((B), dtype=torch.long)
                the positive item ids
            - neg_items: torch.Tensor((B, N), dtype=torch.long)
                the negative item ids

        ## Returns
        loss: torch.Tensor
            the LambdaLoss loss
        """
        user_emb, item_emb, *addition = self.model.embed(norm=self.model.norm)
        user = user_emb[batch.user]                             # (B, emb_size)
        pos_item = item_emb[batch.pos_item]                     # (B, emb_size)
        neg_items = item_emb[batch.neg_items.squeeze(-1)]       # (B, emb_size)
        pos_scores = (user * pos_item).sum(dim=1)               # (B)
        neg_scores = (user * neg_items).sum(dim=1)              # (B)
        d = neg_scores - pos_scores                             # (B)
        pos_ranking = self._cal_ranking(user_emb, item_emb, batch.user, batch.pos_item)                 # (B)
        neg_ranking = self._cal_ranking(user_emb, item_emb, batch.user, batch.neg_items.squeeze(-1))    # (B)
        d_ranking = torch.abs(pos_ranking - neg_ranking)        # (B)
        # delta = torch.abs(1 / torch.log2(1 + d_ranking) - 1 / torch.log2(2 + d_ranking))                # (B)
        delta = 1 / torch.log2(1 + d_ranking)                   # (B)
        loss = (F.softplus(d) * delta).mean()
        loss += self.model.additional_loss(batch, user_emb, item_emb, *addition)
        return loss

