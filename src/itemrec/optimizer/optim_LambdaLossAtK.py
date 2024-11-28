# -------------------------------------------------------------------
# ItemRec / Item Recommendation Benchmark
# Copyright (C) 2024 Tiny Snow / Weiqin Yang @ Zhejiang University
# -------------------------------------------------------------------
# Module: Model - LambdaLoss@K Optimizer
# Description:
#  This module provides the LambdaLoss@K Optimizer for ItemRec. LambdaLoss
#  is a pairwise NDCG optimization algorithm similar to LambdaRank, and
#  LambdaLoss@K is a variant of LambdaLoss that optimizes the top-K ranking
#  performance.
#  - Jagerman, R., Qin, Z., Wang, X., Bendersky, M., & Najork, M. (2022, July). 
#   On optimizing top-k metrics for neural ranking models. 
#   In Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval (pp. 2303-2307).
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
    'LambdaLossAtKOptimizer',
]

# LambdaLossAtKOptimizer --------------------------------------------
class LambdaLossAtKOptimizer(LambdaRankOptimizer):
    r"""
    ## Class
    The LambdaLoss@K Optimizer for ItemRec.
    LambdaLoss is a pairwise NDCG optimization algorithm similar to LambdaRank.
    LambdaLoss@K is a variant of LambdaLoss that optimizes NDCG@K.
    The LambdaLoss@K optimizer is inherited from LambdaRankOptimizer.
    
    ## Algorithms
    The only difference between LambdaLoss and LambdaRank is the definition of
    $\Delta(i, j)$, which is defined as follows:

    $$
    \Delta_{ij} := \delta_{ij}@K \cdot |G(y_i) - G(y_j)|
    $$

    where $G(y_i) = 2^{y_i} - 1$ is the gain of item i ($y_i = 1$ if $(u, i)$ is a positive pair,
    otherwise $y_i = 0$). The $\delta_{ij}$ in LambdaLoss is defined as follows:

    $$
    \delta_{ij}:= \left| \frac{1}{D(|\pi_i - \pi_j|)} - \frac{1}{D(|\pi_i - \pi_j| + 1)} \right|
    $$

    where $D(\pi_i) = \log_2(1 + \pi_i)$ is the discount of item i, $\pi_i$ is the rank of item i.
    However, in LambdaLoss@K, we define $\delta_{ij}@K$ as follows:

    $$
    \delta_{ij}@K = \left\{\begin{aligned}
        & \delta_{ij} \mu_{ij} 	&, & \text{ if } \pi_i > K \text{ or } \pi_j > K	\\
        & \delta_{ij}			&, & \text{ else}
    \end{aligned}\right.
    $$

    The $\mu_{ij}$ is @K decay factor, which is defined as follows:
    
    $$
    \mu_{ij} = \left(1 - \frac{1}{\max(D(\pi_i), D(\pi_j))}\right)^{-1}
    $$
    
    NOTE: For every user u and positive-negative pair (i, j) in a batch, we need to calculate
    the ranking of i and j, with the complexity of O(|U||I|). Thus, LambdaLoss@K is not suitable 
    for large-scale datasets.

    NOTE: Since we only sample one negative item $j_0$ for each positive item, we approximate the
    sum $\sum_{y_i > y_j} \Delta(i, j) \cdot \sigma(d_{uij})$ by the same trick as LambdaLoss.

    ## References
    - Jagerman, R., Qin, Z., Wang, X., Bendersky, M., & Najork, M. (2022, July).
        On optimizing top-k metrics for neural ranking models.
        In Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval (pp. 2303-2307).
    """
    def __init__(self, model: IRModel, lr: float = 0.001, weight_decay: float = 0.0,
        K: int = 10) -> None:
        r"""
        ## Function
        The constructor of the LambdaLoss@K Optimizer.
        
        ## Arguments
        model: IRModel
            the model to be optimized
        lr: float
            the learning rate
        weight_decay: float
            the weight decay parameter
        K: int
            the top-K value for LambdaLoss@K
        """
        super(LambdaLossAtKOptimizer, self).__init__(model, lr, weight_decay)
        self.K = K

    def cal_loss(self, batch: IRDataBatch) -> torch.Tensor:
        r"""
        ## Function
        Calculate the LambdaLoss@K loss for batch data.
        
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
            the LambdaLoss@K loss
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
        max_ranking = torch.max(pos_ranking, neg_ranking)       # (B)
        mu = (1 - 1 / max_ranking).reciprocal()                 # (B)
        mask = max_ranking > self.K                             # (B)
        delta = torch.where(mask, delta * mu, delta)            # (B)
        loss = (F.softplus(d) * delta).mean()
        loss += self.model.additional_loss(batch, user_emb, item_emb, *addition)
        return loss

