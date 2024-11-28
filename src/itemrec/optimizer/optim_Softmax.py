# -------------------------------------------------------------------
# ItemRec / Item Recommendation Benchmark
# Copyright (C) 2024 Tiny Snow / Weiqin Yang @ Zhejiang University
# -------------------------------------------------------------------
# Module: Model - Softmax Optimizer
# Description:
#  This module provides the Softmax (Sampled Softmax Loss) Optimizer for
#  ItemRec. Softmax is a widely used (and de facto standard) loss function
#  for item recommendation. The Softmax optimizer is inherited from IROptimizer.
#  Reference:
#  - Wu, J., Wang, X., Gao, X., Chen, J., Fu, H., Qiu, T., & He, X. (2022). 
#   On the effectiveness of sampled softmax loss for item recommendation. 
#   ACM Transactions on Information Systems.
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
    'SoftmaxOptimizer',
]

# SoftmaxOptimizer --------------------------------------------------
class SoftmaxOptimizer(IROptimizer):
    r"""
    ## Class
    The Softmax (Sampled Softmax Loss) Optimizer for ItemRec.
    Softmax is a widely used (and de facto standard) loss function for item recommendation.
    The Softmax optimizer is inherited from IROptimizer.
    
    ## Algorithms
    The Softmax loss function is defined as:

    $$
    \mathcal{L}_{SL}(u) = \sum_{i \in \mathcal{P}_u} \log \left(\sum_{j \in \mathcal{N}} \exp(d_{uij}) \right) + \lambda \lVert \Theta \rVert^2
    $$

    where $\mathcal{P}_u$ is the set of positive items for user $u$, $d_{uij} = f_{uj} - f_{ui}$, 
    $f_{ui}$ is the score of user $u$ on item $i$, $\lambda$ is the regularization parameter, 
    and $\Theta$ is the model parameters. $\mathcal{N}$ can be seen as the set of all items $\mathcal{I}$,
    $\mathcal{I} \setminus \mathcal{P}_u$, or $(\mathcal{I} \setminus \mathcal{P}_u) \cup \{i\}$, 
    where $i$ is the positive item for user $u$. Here we use the second form, i.e. $\mathcal{N}$ is
    the (sampled) set of negative items for user $u$.

    Note that we may add a temperature parameter $\tau$ to the softmax function to control the
    DRO (Distributional Robustness Optimization)'s trade-off between accuracy and robustness.
    In our implementation, the $tau$ can be applied to $d_{uij}, i.e.

    $$
    \mathcal{L}_{SL}(u) = \sum_{i \in \mathcal{P}_u} \log \left(\sum_{j \in \mathcal{N}} \exp(d_{uij} / \tau) \right) + \lambda \lVert \Theta \rVert^2
    $$

    Some works also add a negative weight to the positive items. However, since it's not make
    sense in the DRO and NDCG-optimization perspective, we do not add this parameter in our 
    implementation.

    NOTE: We use the cosine similarity as the score function, i.e. 
    $f_{ui} = \frac{u \cdot i}{\lVert u \rVert \lVert i \rVert}$.

    NOTE: The temperature parameter can be tuned in the training process if you set
    `adaptive` to True. By default, the temperature is fixed to the given value.
    
    ## References
    - Wu, J., Wang, X., Gao, X., Chen, J., Fu, H., Qiu, T., & He, X. (2022).
        On the effectiveness of sampled softmax loss for item recommendation.
        ACM Transactions on Information Systems.
    """
    def __init__(self, model: IRModel, lr: float = 0.001, weight_decay: float = 0.0, 
        neg_num: int = 1000, tau: float = 1.0) -> None:
        r"""
        ## Function
        The constructor of the Softmax optimizer.
        
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
        """
        super(SoftmaxOptimizer, self).__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.neg_num = neg_num
        self.tau = tau
        params = [{'params': self.model.parameters()}]
        self.optimizer = torch.optim.Adam(
            params,
            lr=self.lr,
            weight_decay=self.weight_decay
        )

    def cal_loss(self, batch: IRDataBatch) -> torch.Tensor:
        r"""
        ## Function
        Calculate the Softmax loss for batch data.
        
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
            the Softmax loss
        """
        user_emb, item_emb, *addition = self.model.embed(norm=self.model.norm)
        user = user_emb[batch.user]                                     # (B, emb_size)
        pos_item = item_emb[batch.pos_item]                             # (B, emb_size)
        neg_items = item_emb[batch.neg_items]                           # (B, N, emb_size)
        pos_scores = F.cosine_similarity(user, pos_item)                # (B)
        neg_scores = F.cosine_similarity(user.unsqueeze(1), neg_items, dim=2)   # (B, N)
        d = neg_scores - pos_scores.unsqueeze(1)                        # (B, N)
        loss = torch.logsumexp(d / self.tau, dim=1).mean()
        loss += self.model.additional_loss(batch, user_emb, item_emb, *addition)
        return loss

