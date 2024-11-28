# -------------------------------------------------------------------
# ItemRec / Item Recommendation Benchmark
# Copyright (C) 2024 Tiny Snow / Weiqin Yang @ Zhejiang University
# -------------------------------------------------------------------
# Module: Model - Bilateral Softmax Loss
# Description:
#  This module provides the AdvInfoNCE Optimizer for ItemRec. AdvInfoNCE 
#  is a novel loss function for item recommendation mainly based on InfoNCE
#  (Contrastive Learning) and adversarial learning. AdvInfoNCE assigns 
#  different hardness to each negative item by adversarial learning.
#  Reference:
#  - Zhang, A., Sheng, L., Cai, Z., Wang, X., & Chua, T. S. (2024). 
#   Empowering Collaborative Filtering with Principled Adversarial Contrastive Loss. 
#   Advances in Neural Information Processing Systems, 36.
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
    'AdvInfoNCEOptimizer',
]

# AdvInfoNCEOptimizer -----------------------------------------------
class AdvInfoNCEOptimizer(IROptimizer):
    r"""
    ## Class
    The AdvInfoNCE (Adversarial InfoNCE) Optimizer for ItemRec.
    AdvInfoNCE is a novel loss function for item recommendation mainly based on InfoNCE
    (Contrastive Learning) and adversarial learning. AdvInfoNCE assigns different hardness
    to each negative item by adversarial learning.

    ## Algorithms
    The AdvInfoNCE introduces the hardness $\delta_{j}^{(u, i)}$ for each positive user-item pair
    $(u, i)$ and negative item $j$. AdvInfoNCE aims to obtain the hardness s.t. for all negative
    items $j$, $s(u, j) - s(u, i) + \delta_{j}^{(u, i)} < 0$, where $s(u, i)$ is the score of $(u, i)$.
    Obviously, the hardness is a $(u, i)$-specific score threshold which controls the feasible zone
    of negative items.

    AdvInfoNCE first defines the hardness $\delta_{j}^{(u, i)}$ as a learnable parameter:

    $$
    \delta_{j}^{(u, i)} = \log(|\mathcal{N}_u| \frac{\exp(g(u, j))}{\sum_{j' \in \mathcal{N}_u} \exp(g(u, j'))})
    $$

    where $g(u, j)$ is just a MF-based function for $(u, j)$ pair, suppose its parameter is $\Theta_{adv}$. 
    Note that in the original implementation, it's in fact user-specific (not user-item pair).

    The AdvInfoNCE loss is defined as:

    $$
    \min_\Theta \mathcal{L}_{\text{advInfoNCE}} = \min_\Theta \max_{\{\delta_{j}^{(u, i)}\} \in \mathbb{C}(\eta)} -\sum_{(u, i) \in \mathcal{O}^+} \log \frac{\exp(s(u, i))}{\exp(s(u, i)) + K\sum_{j \in \mathcal} \exp(\delta_{j}^{(u, i)})\exp(s(u, j))}
    $$

    where $\mathcal{O}^+$ is the set of positive user-item pairs, and $\mathbb{C}(\eta)$ is the set of $\{\delta_{j}^{(u, i)}\}$ constrained in DRO perspective, i.e. $D_{KL}(P(j | u, i) \Vert P_0(j | u, i)) \leq \eta$, where $P_0$ is the uniform negative sampling distribution.
    $K$ is negative weight for balancing the positive and negative samples, default is 64.
    
    In the practical implementation, during each epoch, we will update the
    parameters in two steps:
    - Training model parameters $\Theta$ by minimizing the AdvInfoNCE loss 
        \mathcal{L}_{\text{advInfoNCE}}, the hardness parameters $\Theta_{adv}$ 
        are fixed. 
    - If this epoch should update the hardness, we will update the hardness 
        parameters $\Theta_{adv}$ by maximizing the AdvInfoNCE loss 
        \mathcal{L}_{\text{advInfoNCE}}, the model parameters $\Theta$ are fixed.
        By default, we update the hardness parameters every epoch.
    """
    def __init__(self, model: IRModel, lr: float = 0.001, weight_decay: float = 0.0, 
        neg_num: int = 1000, tau: float = 1.0, neg_weight: float = 64, 
        lr_adv: float = 0.0001, epoch_adv: int = 1) -> None:
        r"""
        ## Function
        The constructor of the AdvInfoNCE optimizer.

        ## Arguments
        model: IRModel
            the model
        lr: float
            the learning rate
        weight_decay: float
            the weight decay
        neg_num: int
            the number of negative samples
        tau: float
            the temperature parameter
        neg_weight: float
            the negative weight
        lr_adv: float
            the learning rate for adversarial learning
        epoch_adv: int
            the epoch interval for adversarial learning
        """
        super(AdvInfoNCEOptimizer, self).__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.neg_num = neg_num
        self.tau = tau
        self.neg_weight = neg_weight
        self.lr_adv = lr_adv
        self.epoch_adv = epoch_adv
        # model optimizer
        self.optimizer_model = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        # adversarial learning MF
        self.adv_user_emb = nn.Embedding(model.user_size, model.emb_size)
        self.adv_item_emb = nn.Embedding(model.item_size, model.emb_size)
        self.adv_user_emb = self.adv_user_emb.to(model.device)
        self.adv_item_emb = self.adv_item_emb.to(model.device)
        # adversarial optimizer
        self.optimizer_adv = torch.optim.Adam(
            [{'params': self.adv_user_emb.parameters()}, {'params': self.adv_item_emb.parameters()}],
            lr=self.lr_adv, 
            weight_decay=self.weight_decay, 
            maximize=True
        )

    def update_adv(self, epoch: int) -> bool:
        r"""
        ## Function
        Whether to update the hardness parameters in this epoch.

        ## Arguments
        epoch: int
            the current epoch

        ## Returns
        bool
            whether to update the hardness parameters
        """
        return (epoch + 1) % self.epoch_adv == 0

    def cal_loss(self, batch: IRDataBatch) -> torch.Tensor:
        r"""
        ## Function
        Calculate the AdvInfoNCE loss for batch data.

        ## Arguments
        batch: IRDataBatch
            the batch data

        ## Returns
        loss: torch.Tensor
            the AdvInfoNCE loss
            if the epoch is for updating the model, return 
        """
        torch.autograd.set_detect_anomaly(True)     # set anomaly detection
        # model embeddings & scores
        user_emb, item_emb, *addition = self.model.embed(norm=self.model.norm)
        user = user_emb[batch.user]                                     # (B, emb_size)
        pos_item = item_emb[batch.pos_item]                             # (B, emb_size)
        neg_items = item_emb[batch.neg_items]                           # (B, N, emb_size)
        pos_scores = F.cosine_similarity(user, pos_item)                # (B)
        neg_scores = F.cosine_similarity(user.unsqueeze(1), neg_items, dim=2)       # (B, N)
        pos_scores = pos_scores / self.tau
        neg_scores = neg_scores / self.tau
        # adversarial embeddings & delta
        adv_user = self.adv_user_emb(batch.user)                        # (B, emb_size)
        adv_neg_items = self.adv_item_emb(batch.neg_items)              # (B, N, emb_size)
        delta = F.cosine_similarity(adv_user.unsqueeze(1), adv_neg_items, dim=2)   # (B, N)
        delta = F.softmax(delta, dim=1)                                 # (B, N), not log here
        delta = delta * self.neg_num                                    # (B, N)
        # AdvInfoNCE loss
        loss = -torch.log(torch.exp(pos_scores) \
            / (torch.exp(pos_scores) + self.neg_weight * torch.sum(delta * torch.exp(neg_scores), dim=1)) \
        ).mean()
        loss += self.model.additional_loss(batch, user_emb, item_emb, *addition)
        return loss

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
        if self.update_adv(epoch):
            self.optimizer_adv.step()       # update the hardness parameters
        else:
            self.optimizer_model.step()     # update the model parameters
        return loss.cpu().item()

    def zero_grad(self) -> None:
        r"""
        ## Function
        Zero the gradients of the optimizer.
        """
        self.optimizer_model.zero_grad()
        self.optimizer_adv.zero_grad()




