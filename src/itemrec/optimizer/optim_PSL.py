# -------------------------------------------------------------------
# ItemRec / Item Recommendation Benchmark
# Copyright (C) 2024 Tiny Snow / Weiqin Yang @ Zhejiang University
# -------------------------------------------------------------------
# Module: Model - Pairwise Softmax Loss
# Description:
#  This module provides the PSL (Pairwise Softmax Loss) Optimizer for ItemRec.
#  PSL is a NDCG surrogate loss function with DRO-robusetness optimization.
#  The PSL optimizer is inherited from IROptimizer.
#  Reference:
#  - Yang, W., Chen, J., Xin, X., Zhou, S., Hu, B., Feng, Y., Chen, C. & Wang, C. 
#   PSL: Rethinking and Improving Softmax Loss from Pairwise Perspective for Recommendation. 
#   In The Thirty-eighth Annual Conference on Neural Information Processing Systems.
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
    'PSLOptimizer',
]

# PSLOptimizer ------------------------------------------------------
class PSLOptimizer(IROptimizer):
    r"""
    ## Class
    The PSL (Pairwise Softmax Loss) Optimizer for ItemRec.
    PSL is a NDCG surrogate loss function with DRO-robusetness optimization.
    The PSL optimizer is inherited from IROptimizer.
    
    ## Algorithms
    The PSL loss function is a Log-Expectation-Exp (LEE) DRO form loss, which 
    is very similar to the Softmax loss function. The DRO object in Softmax loss 
    is $d_{uij}$, where $\tau$ is the temperature parameter. However, in PSL loss
    the DRO object is $\sigma(d_{uij})$, which makes PSL loss better approximating
    NDCG than Softmax loss. 

    By choosing different distributions for DRO, i.e. we can choose the distribution
    $Q$ in the uncertainty set $\mathcal{Q}$ either as $Q(j | u, i)$ or $Q(i, j | u)$,
    we can have two different forms of PSL loss. The first form only considers the
    robustness of the negative items (Softmax-like), while the second form considers
    the robustness of both positive and negative items (BPR-like). You can shoose the
    form by setting `method` to 1 or 2.
    
    $$
    \begin{array}{llll}
        \mathcal{L}_{\textit{PSL}1}(u) : 
            &\mathbb{E}_{i \sim \mathcal{P_u}}[\log \mathbb{E}_{j \sim \mathcal{I}}[\exp(\textcolor{red}{\sigma(d_{uij})} / \tau^*)]] 
            &\sim & 
            \max_{Q\in \mathbb{Q}} \mathbb{E}_{i \sim \mathcal{P_u}}[\mathbb{E}_{j \sim Q(j | u, i)}[\textcolor{red}{\sigma(d_{uij})}]] \\
        \mathcal{L}_{\textit{PSL}2}(u) : 
            &\log\mathbb{E}_{(i, j) \sim \mathcal{P_u} \times \mathcal{I}}[\exp(\textcolor{red}{\sigma(d_{uij})} / \tau^*)] 
            &\sim & 
            \max_{Q\in \mathbb{Q}} \mathbb{E}_{(i, j) \sim Q(i, j | u)}[\textcolor{red}{\sigma(d_{uij})}] \\
    \end{array}
    $$
    
    where $\tau^*$ is the robust temperature determined by the robustness radius 
    in DRO. Note that $\tau^*$ is different from the temperature parameter $\tau$ 
    in the softmax loss. You can think of $\tau$ as the scaling factor of the
    similarity scores, while $\tau^*$ is the scaling factor of the robustness.
    You may refer the notation and algorithms in the `SoftmaxOptimizer`.
    
    The activation function $\sigma$ should satisfy the following inequality: 
    
    $$
    \mathbb{I}(d_{uij} \geq 0) \leq \exp(\sigma(d_{uij})) \leq \exp(d_{uij}), 
    d_{uij} \in [-2 / tau, 2 / tau]
    $$

    For convenience, we assume $\tau \geq 2$, and check the above inequality
    in $d_{uij} \in [-1, 1]$. We provide the following activation functions:
    - 'tanh': $\log (\tanh(x) + 1)$
    - 'relu': $\log (\text{ReLU}(x + 1))$
    - 'atan': $\log (\arctan(x) + 1)$
    You may choose the activation function by the id.

    NOTE: We use the cosine similarity as the score function with temperature, 
    i.e. $f_{ui} = \frac{u \cdot i}{\lVert u \rVert \lVert i \rVert} / \tau$, 
    where $\tau \geq 2$ is the temperature parameter.

    NOTE: The temperature parameter $\tau^*$ can be tuned in the training process
    if you set `adaptive` to True. However, the $\tau$ is fixed, since we need to
    make sure $\tau \geq 2$.
    
    ## References
    - Yang, W., Chen, J., Xin, X., Zhou, S., Hu, B., Feng, Y., Chen, C. & Wang, C.
        PSL: Rethinking and Improving Softmax Loss from Pairwise Perspective for Recommendation.
        In The Thirty-eighth Annual Conference on Neural Information Processing Systems.
    """
    def __init__(self, model: IRModel, lr: float = 0.001, weight_decay: float = 0.0, 
        neg_num: int = 1000, tau: float = 1.0, tau_star: float = 1.0, method: int = 1, 
        activation: str = 'tanh') -> None:
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
            the temperature parameter for the score function
        tau_star: float
            the temperature parameter for the robustness
        method: int
            the id of the PSL method, must be one of [1, 2]
        activation: str
            the id of the activation function, must be one of
            ['tanh', 'relu', 'atan']
        """
        super(PSLOptimizer, self).__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.neg_num = neg_num
        self.tau = tau
        self.tau_star = tau_star
        self.method = method
        assert method in [1, 2], f"Invalid method for PSL: {method}, must be one of [1, 2]"
        if activation == 'tanh':
            self.sigma = lambda x: torch.log(torch.tanh(x) + 1)
        elif activation == 'relu':
            self.sigma = lambda x: torch.log(F.relu(x + 1))
        elif activation == 'atan':
            self.sigma = lambda x: torch.log(torch.atan(x) + 1)
        else:
            raise ValueError(f"Invalid activation function for PSL: {activation}, must be one of ['tanh', 'relu', 'atan']")
        params = [{'params': self.model.parameters()}]
        self.optimizer = torch.optim.Adam(
            params, 
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

    def cal_loss(self, batch: IRDataBatch) -> torch.Tensor:
        r"""
        ## Function
        Calculate the PSL loss for the given batch of data.
        
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
            the PSL loss
        """
        user_emb, item_emb, *addition = self.model.embed(norm=self.model.norm)
        user = user_emb[batch.user]                                     # (B, emb_size)
        pos_item = item_emb[batch.pos_item]                             # (B, emb_size)
        neg_items = item_emb[batch.neg_items]                           # (B, N, emb_size)
        pos_scores = F.cosine_similarity(user, pos_item) / self.tau     # (B)
        neg_scores = F.cosine_similarity(user.unsqueeze(1), neg_items, dim=2) / self.tau    # (B, N)
        d = neg_scores - pos_scores.unsqueeze(1)                        # (B, N)
        if self.method == 1:        # PSL1: softmax-like
            loss = torch.logsumexp(self.sigma(d) / self.tau_star, dim=1).mean()
        elif self.method == 2:      # PSL2: BPR-like
            loss = torch.exp(self.sigma(d) / self.tau_star).mean()
            loss = torch.log(loss)
        loss += self.model.additional_loss(batch, user_emb, item_emb, *addition)
        return loss

