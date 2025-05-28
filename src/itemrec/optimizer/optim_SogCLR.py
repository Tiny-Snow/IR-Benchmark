# -------------------------------------------------------------------
# ItemRec / Item Recommendation Benchmark
# Copyright (C) 2024 Tiny Snow / Weiqin Yang @ Zhejiang University
# -------------------------------------------------------------------
# Module: Model - SogCLR Optimizer
# Description:
#  This module provides the SogCLR (Stochastic Optimization algorithm for 
#  solving the Global objective of Contrastive Learning of Representations)
#  ItemRec. SogCLR aims to achieve a good trade-off between the accuracy and
#  the negative sampling size. The SogCLR optimizer is inherited from IROptimizer.
#  Reference:
#  - Yuan, Z., Wu, Y., Qiu, Z. H., Du, X., Zhang, L., Zhou, D., & Yang, T. (2022, June). 
#   Provable stochastic optimization for global contrastive learning: Small batch does not harm performance. 
#   In International Conference on Machine Learning (pp. 25760-25782). PMLR.
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
    'SogCLROptimizer',
]

# SogCLROptimizer ---------------------------------------------------
class SogCLROptimizer(IROptimizer):
    r"""
    ## Class
    The SogCLR (Stochastic Optimization algorithm for solving the Global objective of
    Contrastive Learning of Representations) Optimizer for ItemRec.
    SogCLR aims to achieve a good trade-off between the accuracy and the negative sampling size.
    The SogCLR optimizer is inherited from IROptimizer.
    
    ## Algorithms
    The SogCLR adopts the Compositional Optimization algorithm to the Softmax Loss. Specifically,
    the gradient is:
    
    $$
    \nabla \mathcal{L}_{\textnormal{SL}}(u) = \underbrace{-\sum_{i \in \mathcal{P}_u} \nabla s_{ui}^+ / \tau}_{\text{positive gradient}} + \underbrace{\frac{|\mathcal{P}_u|}{|\mathcal{N}_{u}| g_u} \sum_{j \in \mathcal{N}_u} \nabla \exp(s_{uj}^- / \tau)}_{\text{negative gradient}} ,
    $$
    
    where $\mathcal{P}_u$ is the set of positive items for user $u$, $\mathcal{N}_u$ is the set of 
    negative items for user $u$, $s_{ui}^+$ is the positive score, $s_{uj}^-$ is the negative score,
    $g_u = \mathbb{E}_{j \sim U(j | u)} \left[ \exp(s_{uj}^- / \tau) \right]$, $U(j | u)$ is the 
    uniform distribution over $\mathcal{N}_u$, and $\tau$ is the temperature parameter.
    
    In SogCLR, the $g_u$ in the denominator is replaced by a moving average estimator $g_{u}^{(t)}$,
    which is updated by:
    
    $$
    g_{u}^{(t)} = (1 - \gamma_g) g_{u}^{(t - 1)} + \frac{\gamma_g}{|\mathcal{\hat{\mathcal{N}}_u}|} \sum_{j \in \hat{\mathcal{N}}_u} \exp(s_{uj}^- / \tau)
    $$
    
    where $\gamma_g$ is the hyperparameter.

    ## References
    - Yuan, Z., Wu, Y., Qiu, Z. H., Du, X., Zhang, L., Zhou, D., & Yang, T. (2022, June).
        Provable stochastic optimization for global contrastive learning: Small batch does not harm performance.
        In International Conference on Machine Learning (pp. 25760-25782). PMLR.
    """
    def __init__(self, model: IRModel, lr: float = 0.001, weight_decay: float = 0.0, 
        neg_num: int = 1000, tau: float = 1.0, gamma_g: float = 0.9) -> None:
        r"""
        ## Function
        The constructor of the SogCLROptimizer.
        
        ## Arguments
        model: IRModel
            the model to be optimized
        lr: float
            the learning rate
        weight_decay: float
            the weight decay parameter
        neg_num: int
            the number of negative items for each user
        tau: float  
            the temperature parameter for the softmax function
        gamma_g: float
            the hyperparameter for the moving average estimator
        """
        super(SogCLROptimizer, self).__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.neg_num = neg_num
        self.tau = tau
        self.gamma_g = gamma_g
        self.g = torch.zeros((model.user_size), dtype=torch.float32).to(model.device)
        params = [{'params': self.model.parameters()}]
        self.optimizer = torch.optim.Adam(
            params,
            lr=self.lr,
            weight_decay=self.weight_decay
        )

    def cal_loss(self, batch: IRDataBatch) -> torch.Tensor:
        r"""
        ## Function
        Calculate the SogCLR loss for batch data.
        
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
            the SogCLR loss
        """
        user_emb, item_emb, *addition = self.model.embed(norm=self.model.norm)
        user = user_emb[batch.user]                                     # (B, emb_size)
        pos_item = item_emb[batch.pos_item]                             # (B, emb_size)
        neg_items = item_emb[batch.neg_items]                           # (B, N, emb_size)
        pos_scores = F.cosine_similarity(user, pos_item)                # (B)
        neg_scores = F.cosine_similarity(user.unsqueeze(1), neg_items, dim=2)   # (B, N)
        g_hat = torch.exp(neg_scores / self.tau).mean(dim=1)            # (B)
        g = self.g[batch.user]                                          # (B)
        g = torch.where(g == 0, g_hat.detach(), g)                      # (B)
        g = (1 - self.gamma_g) * g + self.gamma_g * g_hat.detach()      # (B)
        self.g[batch.user] = g
        loss = -(pos_scores / self.tau).mean() + (g_hat / g).mean()
        loss += self.model.additional_loss(batch, user_emb, item_emb, *addition)
        return loss

