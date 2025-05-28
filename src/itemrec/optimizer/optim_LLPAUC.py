# -------------------------------------------------------------------
# ItemRec / Item Recommendation Benchmark
# Copyright (C) 2024 Tiny Snow / Weiqin Yang @ Zhejiang University
# -------------------------------------------------------------------
# Module: Model - LLPAUC (Lower-Left Partial AUC)
# Description:
#  This module provides the LLPAUC Optimizer for ItemRec. LLPAUC is a partial AUC
#  loss function for item recommendation, which focuses on the lower-left part of
#  the ROC curve so as to optimize Top-K metrics.
#  Reference:
#  - Shi, W., Wang, C., Feng, F., Zhang, Y., Wang, W., Wu, J., & He, X. (2024). 
#   Lower-Left Partial AUC: An Effective and Efficient Optimization Metric for Recommendation. 
#   arXiv preprint arXiv:2403.00844.
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
    'LLPAUCOptimizer',
]

# LLPAUCOptimizer ----------------------------------------------------
class LLPAUCOptimizer(IROptimizer):
    r"""
    ## Class
    The LLPAUC (Lower-Left Partial AUC) Optimizer for ItemRec.
    LLPAUC is a partial AUC loss function for item recommendation, which focuses 
    on the lower-left part of the ROC curve so as to optimize Top-K metrics.

    ## Algorithms
    We know that the AUC (Area Under the Curve) is

    $$
    \text{AUC}(u) = \mathbb{P}_{(i \in \mathcal{P}_u, j \notin \mathcal{P}_u)}[f_{ui} > f_{uj}]
    $$

    where $\mathcal{P}_u$ is the set of positive items for user $u$, and $f_{ui}$ is the
    predicted score of item $i$ for user $u$. 
    
    The LLPAUC only considers the lower-left part of the ROC curve, i.e. $TPR \leq \alpha$ and
    $FPR \leq \beta$. Thus, only top items are considered. The LLPAUC is defined as

    $$
    \text{LLPAUC}(u, \alpha, \beta) = \mathbb{P}_{(i \sim \mathcal{P}_u, j \sim \mathcal{I} \setminus \mathcal{P}_u)} [f_{ui} > f_{uj} \land f_{ui} \geq \eta_\alpha \land f_{uj} \leq \eta_\beta]
    $$

    where $\eta_\alpha$ and $\eta_\beta$ satisfy
    
    $$
    \mathbb{P}_{i \sim \mathcal{P}_u}[f_{ui} \geq \eta_\alpha] = \alpha, \quad
    \mathbb{P}_{j \sim \mathcal{I} \setminus \mathcal{P}_u}[f_{uj} \leq \eta_\beta] = \beta
    $$

    The LLPAUC should be replcaed by a surrogate loss function for optimization.
    By an average Top-K trick and some approximation, the final LLPAUC loss is
    
    $$
    \mathcal{L}_\text{LLPAUC}(u, \alpha, \beta) = \min_{\{\theta, (a, b) \in [0, 1]^{2}, s^{-} \in \mathbb{R}\}} \max_{\{\gamma \in \Omega_{\gamma}, s^{+} \in \mathbb{R}\}} \sum_{i \in \mathcal{P}_{u}} \frac{-\alpha s^{+} - r_{\kappa}(-\ell_{+}(f_{ui}) - s^{+})}{n_{u}^{+}} + \sum_{j \notin \mathcal{P}_{u}}\frac{\beta s^{-} + r_{\kappa}(\ell_{-}(f_{uj}) - s^{-})}{n_{u}^{-}} - (w + 1)\gamma^2
    $$

    where
    - $\Omega_{\gamma} = [\max(-a, b - 1), 1]$
    - $\ell_{+}(f_{ui}) = (f_{ui} - a)^2 - 2(1 + \gamma)f_{ui}$, $\ell_{-}(f_{uj}) = (f_{uj} - b)^2 + 2(1 + \gamma)f_{uj}$. 
    - $r_{\kappa}(x) = \frac{1}{\kappa}\log(1 + \exp(\kappa x))$, i.e. softplus function, 
        which used to approximate the $[\cdot]_{+}$
    - $w$ is used to ensure the strong concavity of the loss w.r.t. $\gamma$, 
        $w > 4\kappa$ 
    - $n_{u}^{+}$ and $n_{u}^{-}$ are the number of positive and negative items for user $u$.
        
    LLPAUC is a min-max optimization problem, the paper solves it by SGDA 
    (Stochastic Gradient Descent Ascent): 
    ```
    Parameters: tau = {theta, a, b, s-}, tau' = {s+, gamma}
    for t = 0, 1, ..., T do
        Sample a mini-batch positive interaction B+
        Uniformly sample a mini-batch negative interaction B- for each positive interaction in B+
        Compute loss F(tau_{t}, tau'_{t}) on B+ and B- and its gradient \nabla F(tau, tau')
        Update tau_{t+1} = \tau_{t} - \eta \nabla F(tau, tau')
        Update tau'_{t+1} = \tau'_{t} + \eta \nabla F(tau, tau')
        Update \tau_{t+1} = Clip(\tau_{t+1})
        Update \tau'_{t+1} = Clip(\tau'_{t+1})
    end for
    Return tau_{T}
    ```
    We use Adam optimizer to solve the LLPAUC optimization problem.
    We simply set two Adam optimizers for $\tau$ and $\tau'$ respectively.
    
    The initialization and clip range of the above parameters is (following the original paper):
    - $a = 1.0, a \in [0, 1]$
    - $b = 0.0, b \in [0, 1]$
    - $s^{+} = 0.5, s^{+} \in [-1, 4]$
    - $s^{-} = 0.5, s^{-} \in [0, 5]$
    - $\gamma = 0.0, \gamma \in [-1, 1]$
    - $\kappa = 5$
    NOTE: For the constraint of $\Omega_{\gamma}$, the original code adds a penalty term 
        $\theta_a (a + \gamma) + \theta_b (\gamma + 1 - b)$.
    NOTE: In the original implementation, the $w$ is not included.

    For the original LLPAUC implementation, please refer to: 
    https://github.com/swt-user/LLPAUC/blob/main/recbole/model/general_recommender/bpr_tp_point.py
    https://github.com/swt-user/LLPAUC/blob/main/recbole/trainer/trainer.py
    
    ## References
    - Shi, W., Wang, C., Feng, F., Zhang, Y., Wang, W., Wu, J., & He, X. (2024).
        Lower-Left Partial AUC: An Effective and Efficient Optimization Metric for Recommendation.
        arXiv preprint arXiv:2403.00844.
    """
    def __init__(self, model: IRModel, lr: float = 0.001, weight_decay: float = 0.0, 
        neg_num: int = 1000, alpha: float = 0.7, beta: float = 0.1) -> None:
        r"""
        ## Function
        The constructor of LLPAUC optimizer.

        ## Arguments
        model: IRModel
            the model to be optimized
        lr: float
            the learning rate
        weight_decay: float
            the weight decay parameter
        neg_num: int
            the number of negative items for each user
        alpha: float
            the $\alpha$ parameter of LLPAUC, default is 0.7
        beta: float
            the $\beta$ parameter of LLPAUC, default is 0.1
        """
        super(LLPAUCOptimizer, self).__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.neg_num = neg_num
        self.alpha = alpha
        self.beta = beta
        # set additional parameters
        self.a = nn.Parameter(torch.tensor(1.0))        # a
        self.b = nn.Parameter(torch.tensor(0.0))        # b
        self.sp = nn.Parameter(torch.tensor(0.5))       # s+
        self.sn = nn.Parameter(torch.tensor(0.5))       # s-
        self.gamma = nn.Parameter(torch.tensor(0.0))    # gamma
        self.kappa = 5                                  # kappa for softplus
        self.theta_a = nn.Parameter(torch.tensor(0.5)) # lambda_a
        self.theta_b = nn.Parameter(torch.tensor(0.5)) # lambda_b
        # set optimizer, we completely follow the original implementation
        self.optimizer_descent = torch.optim.Adam([
            {'params': self.model.parameters(), 'lr': lr, 'weight_decay': weight_decay},
            {'params': [self.a], 'lr': lr, 'weight_decay': weight_decay},
            {'params': [self.b], 'lr': lr, 'weight_decay': weight_decay},
            {'params': [self.sn], 'lr': 2 * lr, 'weight_decay': weight_decay}, 
            {'params': [self.theta_a], 'lr': lr, 'weight_decay': weight_decay},
            {'params': [self.theta_b], 'lr': lr, 'weight_decay': weight_decay},
        ], maximize=False)
        self.optimizer_ascent = torch.optim.Adam([
            {'params': [self.sp], 'lr': 2 * lr, 'weight_decay': weight_decay},
            {'params': [self.gamma], 'lr': 2 * lr, 'weight_decay': weight_decay},
        ], maximize=False)      # NOTE: if ascent here, the effect was a disaster!
        # NOTE: initialize the model parameters
        nn.init.normal_(self.model.user_emb.weight.data, mean=0, std=0.01)
        nn.init.normal_(self.model.item_emb.weight.data, mean=0, std=0.01)

    def _clip(self) -> None:
        r"""
        ## Function
        Clip the parameters to their valid ranges.
        """
        self.a.data.clamp_(0, 1)
        self.b.data.clamp_(0, 1)
        self.sp.data.clamp_(-1, 4)
        self.sn.data.clamp_(0, 5)
        a, b = self.a.item(), self.b.item()
        self.gamma.data.clamp_(max(-a, b - 1), 1)

    def cal_loss(self, batch: IRDataBatch) -> torch.Tensor:
        r"""
        ## Function
        Calculate the LLPAUC loss for batch data.

        ## Arguments
        batch: IRDataBatch
            the batch data

        ## Returns
        The loss of the batch data.
        """
        user_emb, item_emb, *addition = self.model.embed(norm=self.model.norm)
        user = user_emb[batch.user]                                     # (B, emb_size)
        pos_item = item_emb[batch.pos_item]                             # (B, emb_size)
        neg_items = item_emb[batch.neg_items]                           # (B, N, emb_size)
        # NOTE: LLPAUC uses sigmoid similarity
        pos_scores = torch.sigmoid((user * pos_item).sum(dim=1))         # (B)
        neg_scores = torch.sigmoid((user.unsqueeze(1) * neg_items).sum(dim=2)) # (B, N)
        scores_p = -(pos_scores - self.a).square() + 2 * (1 + self.gamma) * pos_scores - self.sp # (B)
        scores_n = (neg_scores - self.b).square() + 2 * (1 + self.gamma) * neg_scores - self.sn # (B, N)
        scores_p = F.softplus(scores_p, beta=self.kappa).mean()
        scores_n = F.softplus(scores_n, beta=self.kappa).mean()
        # here we follow the original implementation, but is that correct?
        loss = -self.sp - scores_p / self.alpha + self.sn + scores_n / self.beta - self.gamma.square()
        loss += self.theta_a * (self.a + self.gamma) + self.theta_b * (self.gamma + 1 - self.b)
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
        self.optimizer_descent.step()
        self.optimizer_ascent.step()
        self._clip()
        return loss.cpu().item()
    
    def zero_grad(self) -> None:
        r"""
        ## Function
        Zero the gradients of the optimizer.
        """
        self.optimizer_descent.zero_grad()
        self.optimizer_ascent.zero_grad()

