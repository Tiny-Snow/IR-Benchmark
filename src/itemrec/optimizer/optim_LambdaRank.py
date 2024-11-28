# -------------------------------------------------------------------
# ItemRec / Item Recommendation Benchmark
# Copyright (C) 2024 Tiny Snow / Weiqin Yang @ Zhejiang University
# -------------------------------------------------------------------
# Module: Model - LambdaRank Optimizer
# Description:
#  This module provides the LambdaRank Optimizer for ItemRec. LambdaRank
#  is a famous pairwise NDCG optimization algorithm.
#  - Burges, C. J. (2010). 
#   From ranknet to lambdarank to lambdamart: An overview. 
#   Learning, 11(23-581), 81.
#  - Burges, C., Ragno, R., & Le, Q. (2006). 
#   Learning to rank with nonsmooth cost functions. 
#   Advances in neural information processing systems, 19.
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

# public functions --------------------------------------------------
__all__ = [
    'LambdaRankOptimizer',
]

# LambdaRankOptimizer -----------------------------------------------
class LambdaRankOptimizer(IROptimizer):
    r"""
    ## Class
    The LambdaRank Optimizer for ItemRec.
    LambdaRank is a famous pairwise NDCG optimization algorithm.
    The LambdaRank optimizer is inherited from IROptimizer.
    
    ## Algorithms
    The LambdaRank loss is defined as follows:

    $$
    \mathcal{L}_{LambdaRank}(u) = \sum_{y_i > y_j} \Delta(i, j) \cdot \sigma(d_{uij}) + \lambda \lVert \Theta \rVert^2
    $$

    where $d_{uij} = f_{uj} - f_{ui}$, the activation function $\sigma$ is the Softplus function, 
    i.e., $\sigma(d) = \log(1 + \exp(d))$. $y_i$ is the label of item i, and $y_i > y_j$ means
    item pair (i, j) is a positive-negative pair. $\Theta$ is the parameters of the model, and 
    $\lambda$ is the weight decay parameter. $\Delta(i, j)$ is the absolute difference of NDCG values 
    when the items $i$ and $j$ are swapped, i.e., 

    $$
    \Delta(i, j) = |DCG(i) - DCG(j)| = |G(y_i) - G(y_j)||1 / D(\pi_i) - 1 / D(\pi_j)|
    $$

    where $G(y_i) = 2^{y_i} - 1$ is the gain of item i, and $D(\pi_i) = \log_2(1 + \pi_i)$ 
    is the discount of item i, where $\pi_i$ is the rank of item i. Since (i, j) is a 
    positive-negative pair, $|G(y_i) - G(y_j)|$ will always be $1$. Therefore, LambdaRank
    is in fact a weighted BPR loss, with the weight $|1 / D(\pi_i) - 1 / D(\pi_j)|$.

    NOTE: For every user u and positive-negative pair (i, j) in a batch, we need to calculate
    the ranking of i and j, with the complexity of O(|U||I|). Thus, LambdaRank is not suitable 
    for large-scale datasets.
    
    ## References
    - Burges, C. J. (2010).
        From ranknet to lambdarank to lambdamart: An overview.
        Learning, 11(23-581), 81.
    - Burges, C., Ragno, R., & Le, Q. (2006).
        Learning to rank with nonsmooth cost functions.
        Advances in neural information processing systems, 19.
    """
    def __init__(self, model: IRModel, lr: float = 0.001, weight_decay: float = 0.0) -> None:
        r"""
        ## Function
        The constructor of the LambdaRank Optimizer.
        
        ## Arguments
        model: IRModel
            the model to be optimized
        lr: float
            the learning rate
        weight_decay: float
            the weight decay parameter
        """
        super(LambdaRankOptimizer, self).__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

    def cal_loss(self, batch: IRDataBatch) -> torch.Tensor:
        r"""
        ## Function
        Calculate the LambdaRank loss for batch data.
        
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
            the LambdaRank loss
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
        delta = torch.abs(1 / torch.log2(1 + pos_ranking) - 1 / torch.log2(1 + neg_ranking))            # (B)
        loss = (F.softplus(d) * delta).mean()
        loss += self.model.additional_loss(batch, user_emb, item_emb, *addition)
        return loss

    def _cal_ranking(self, user_emb: torch.Tensor, item_emb: torch.Tensor, 
        user: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        r"""
        ## Function
        Calculate the ranking of the given item for the user.

        ## Arguments
        user_emb: torch.Tensor
            the user embeddings, with shape (U, emb_size)
        item_emb: torch.Tensor
            the item embeddings, with shape (I, emb_size)
        user: torch.Tensor
            the user ids, with shape (B)
        item: torch.Tensor
            the item ids, with shape (B)
        
        ## Returns
        ranking: torch.Tensor
            the ranking of the items, with shape (B)
        """
        user = user_emb[user]                                   # (B, emb_size)
        scores = user @ item_emb.T                              # (B, I)
        item_scores = scores[torch.arange(len(user)), item]     # (B)
        ranking = (scores > item_scores.unsqueeze(-1)).sum(dim=1) + 1   # (B)
        return ranking

