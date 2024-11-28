# -------------------------------------------------------------------
# ItemRec / Item Recommendation Benchmark
# Copyright (C) 2024 Tiny Snow / Weiqin Yang @ Zhejiang University
# -------------------------------------------------------------------
# Module: Model - Bilateral Softmax Loss
# Description:
#  This module provides the BSL (Bilateral Softmax Loss) Optimizer for 
#  ItemRec. BSL is a novel loss function for item recommendation, which
#  considers the bilateral robustness of both positive and negative items.
#  Reference:
#  - Wu, J., Chen, J., Wu, J., Shi, W., Zhang, J., & Wang, X. (2023). 
#   BSL: Understanding and Improving Softmax Loss for Recommendation. 
#   arXiv preprint arXiv:2312.12882.
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
from .optim_Base import IROptimizer
from ..dataset import IRDataBatch
from ..model import IRModel

# public functions --------------------------------------------------
__all__ = [
    'BSLOptimizer',
]

# BSLOptimizer ------------------------------------------------------
class BSLOptimizer(IROptimizer):
    r"""
    ## Class
    The BSL (Bilateral Softmax Loss) Optimizer for ItemRec.
    BSL is a novel loss function for item recommendation, which considers the bilateral
    robustness of both positive and negative items.

    ## Algorithms
    The BSL loss pseudo code is as follows (we correct the original code in the paper):

    ```python
    # f: user and item embedding table
    # t1: temperature scaling for positive samples
    # t2: temperature scaling for negative samples
    for (u, i, j) in loader:
    # load a minibatch (u,i) with m negative samples
    # dimension u: [B]; i:[B]; j:[B,m]
    emb_u, emb_i, emb_j = f(u), f(i), f(j)
    # dimension u: [B, D]; i:[B, D]; j:[B, m, D]
    L = loss_fn(emb_u, emb_i, emb_j)
    L.backward() # back-propagate
    update(f) # Adam update

    def loss_fn(emb_u, emb_i, emb_j):
    emb_u = normalize(emb_u, dim=1) # l2-normalize
    emb_i = normalize(emb_i, dim=1) # l2-normalize
    emb_j = normalize(emb_j, dim=1) # l2-normalize
    pos_score=(emb_u * emb_i).sum(dim=1) # dimension: [B]
    neg_score=(emb_u.unsqueeze(1) * emb_j).sum(dim=2)
    # dimension: [B, m]
    L(BSL) = -((pos_score/t1).exp() / (neg_score/t2).exp().sum(dim=1).pow(t2/t1)).log()
    return L(BSL)
    ```

    We follow the above pseudo code to implement the BSL loss in PyTorch.
    You may also refer to the original implementation in the paper:
    https://github.com/junkangwu/BSL/blob/master/utils/losses.py

    ## References
    - Wu, J., Chen, J., Wu, J., Shi, W., Zhang, J., & Wang, X. (2023).
        BSL: Understanding and Improving Softmax Loss for Recommendation.
        arXiv preprint arXiv:2312.12882.
    """
    def __init__(self, model: IRModel, lr: float = 0.001, weight_decay: float = 0.0,
        neg_num: int = 1000, tau1: float = 1.0, tau2: float = 1.0) -> None:
        r"""
        ## Function
        Initialize the BSL optimizer.

        ## Arguments
        - model: IRModel
            The model to be optimized
        - lr: float
            The learning rate
        - weight_decay: float
            The weight decay parameter
        - neg_num: int
            The number of negative items for each user
        - tau1: float
            The temperature parameter for the positive items
        - tau2: float
            The temperature parameter for the negative items
        """
        super(BSLOptimizer, self).__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.neg_num = neg_num
        self.tau1 = tau1
        self.tau2 = tau2
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

    def cal_loss(self, batch: IRDataBatch) -> torch.Tensor:
        r"""
        ## Function
        Calculate the BSL loss for batch data.

        ## Arguments
        - batch: IRDataBatch
            The batch data, with shapes:
            - user: torch.Tensor((B), dtype=torch.long)
                The user ids
            - pos_item: torch.Tensor((B), dtype=torch.long)
                The positive item ids
            - neg_items: torch.Tensor((B, N), dtype=torch.long)
                The negative item ids

        ## Returns
        - loss: torch.Tensor
            The BSL loss
        """
        user_emb, item_emb, *addition = self.model.embed(norm=self.model.norm)
        user = user_emb[batch.user]                 # (B, emb_size)
        pos_item = item_emb[batch.pos_item]         # (B, emb_size)
        neg_items = item_emb[batch.neg_items]       # (B, N, emb_size)
        pos_scores = F.cosine_similarity(user, pos_item)                                    # (B)
        neg_scores = F.cosine_similarity(user.unsqueeze(1), neg_items, dim=2)               # (B, N)
        pos_scores = (pos_scores / self.tau1).exp()                                         # (B)
        neg_scores = (neg_scores / self.tau2).exp().sum(dim=1).pow(self.tau2 / self.tau1)   # (B)
        loss = -torch.log(pos_scores / neg_scores).mean()
        loss += self.model.additional_loss(batch, user_emb, item_emb, *addition)
        return loss

