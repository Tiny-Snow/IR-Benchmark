# -------------------------------------------------------------------
# ItemRec / Item Recommendation Benchmark
# Copyright (C) 2024 Tiny Snow / Weiqin Yang @ Zhejiang University
# -------------------------------------------------------------------
# Module: Model - BPR Optimizer
# Description:
#  This module provides the BPR (Bayesian Personalized Ranking) Optimizer
#  for ItemRec. BPR is a pairwise loss function, which is widely used in
#  recommendation systems. The BPR optimizer is inherited from IROptimizer.
#  Reference:
#  - Rendle, S., Freudenthaler, C., Gantner, Z., & Schmidt-Thieme, L. (2012). 
#   BPR: Bayesian personalized ranking from implicit feedback. 
#   arXiv preprint arXiv:1205.2618.
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
    'BPROptimizer',
]

# BPROptimizer ------------------------------------------------------
class BPROptimizer(IROptimizer):
    r"""
    ## Class
    The BPR (Bayesian Personalized Ranking) Optimizer for ItemRec.
    BPR is a pairwise loss function, which is widely used in recommendation systems.
    The BPR optimizer is inherited from IROptimizer.
    
    ## Algorithms
    The BPR loss function is defined as:

    $$
    \mathcal{L}_{BPR}(u) = \sum_{i \in \mathcal{P}_u} \sum_{j \notin \mathcal{P}_u} 
    \log \sigma(f_{ui} - f_{uj}) + \lambda \lVert \Theta \rVert^2
    $$

    where $\mathcal{P}_u$ is the set of positive items for user $u$, $\sigma$ is the 
    sigmoid function, $f_{ui}$ is the score of user $u$ on item $i$, $\lambda$ is the
    regularization parameter, and $\Theta$ is the model parameters.

    We use $softplus$ as the activation function to simplify the calculation.

    NOTE: We use the dot product similarity as the score function, i.e. 
    $f_{ui} = \frac{u \cdot i}{\lVert u \rVert \lVert i \rVert}$.

    ## References
    - Rendle, S., Freudenthaler, C., Gantner, Z., & Schmidt-Thieme, L. (2012).
        BPR: Bayesian personalized ranking from implicit feedback.
        arXiv preprint arXiv:1205.2618.
    """
    def __init__(self, model: IRModel, lr: float = 0.001, weight_decay: float = 0.0) -> None:
        r"""
        ## Function
        The constructor of the BPR optimizer.
        
        ## Arguments
        model: IRModel
            the model to be optimized
        lr: float
            the learning rate
        weight_decay: float
            the weight decay parameter
        """
        super(BPROptimizer, self).__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        self._init_weights()

    def _init_weights(self):
        r"""
        ## Function
        Initialize the weights of the model.
        Note that BPR needs a smaller variance for the initialization.
        """
        nn.init.xavier_normal_(self.model.user_emb.weight)
        nn.init.xavier_normal_(self.model.item_emb.weight)

    def cal_loss(self, batch: IRDataBatch) -> torch.Tensor:
        r"""
        ## Function
        Calculate the BPR loss for batch data.
        
        ## Arguments
        batch: IRDataBatch
            the batch data, with shapes:
            - user: torch.Tensor((B), dtype=torch.long)
                the user ids
            - pos_item: torch.Tensor((B), dtype=torch.long)
                the positive item ids
            - neg_items: torch.Tensor((B, 1), dtype=torch.long)
                the negative item ids
    
        ## Returns
        loss: torch.Tensor
            the BPR loss
        """
        user_emb, item_emb, *addition = self.model.embed(norm=self.model.norm)
        user = user_emb[batch.user]                             # (B, emb_size)
        pos_item = item_emb[batch.pos_item]                     # (B, emb_size)
        neg_items = item_emb[batch.neg_items.squeeze(-1)]       # (B, emb_size)
        pos_scores = (user * pos_item).sum(dim=1)               # (B)
        neg_scores = (user * neg_items).sum(dim=1)              # (B)
        d = neg_scores - pos_scores
        loss = F.softplus(d).mean()
        loss += self.model.additional_loss(batch, user_emb, item_emb, *addition)
        return loss

