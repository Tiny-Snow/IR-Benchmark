# -------------------------------------------------------------------
# ItemRec / Item Recommendation Benchmark
# Copyright (C) 2024 Tiny Snow / Weiqin Yang @ Zhejiang University
# -------------------------------------------------------------------
# Module: Model - GuidedRec
# Description:
#  This module provides the GuidedRec Optimizer for ItemRec. GuidedRec is a 
#  parameterized DCG surrogate loss method, which using BCE loss as the
#  main loss, while minimizing the difference between the estimated DCG and
#  the true DCG. 
#  Reference:
#  - Rashed, A., Grabocka, J., & Schmidt-Thieme, L. (2021, July). 
#   A guided learning approach for item recommendation via surrogate loss learning. 
#   In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval (pp. 605-613).
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
    'GuidedRecOptimizer',
]

# GuidedRec Optimizer ----------------------------------------------
class GuidedRecOptimizer(IROptimizer):
    r"""
    ## Class
    The GuidedRec Optimizer for ItemRec.
    GuidedRec is a parameterized DCG surrogate loss method, which using BCE loss as the
    main loss, while minimizing the difference between the estimated DCG and the true DCG.

    ## Algorithms

    GuidedRec uses three loss functions to optimize the model:
    
    1. Backbone Loss: BCE Loss (Log Loss in the original paper)

    In a batch of data, the BCE loss is similar to the pairwise loss, which only samples 
    one positive item and one negative item for each user. The BCE loss is calculated as:

    $$
    \mathcal{L}_{\text{BCE}} = -\sum_{u \in \mathcal{U}, i \in \mathcal{P}_u, j \notin \mathcal{P}_u} \log(\sigma(f(u, i))) + \log(1 - \sigma(f(u, j)))
    $$

    where $\sigma$ is the sigmoid function, $\mathcal{U}$ is the set of users, $\mathcal{P}_u$ 
    is the set of positive items for user $u$, and $f(u, i)$ is the score of user $u$ on item $i$.

    In practice, the ratio of positive and negative items is set to 1:1. Thus, if we sample $N - 1$ 
    negative items for each user, the contribution of the positive item is multiplied by $N - 1$.
    In the original paper, $N = 10$.

    2. Surrogate Loss: Minimize the difference between the estimated DCG and the true DCG

    GuidedRec calculates the estimated DCG by parameterized method. The detailed calculation
    can be found in our code below. The loss function is defined as:

    $$
    \mathcal{L}_{\text{Surrogate}} = \sum_{u \in \mathcal{U}} ||\text{EDCG}(u) - \text{DCG}(u)||_2^2
    $$

    3. DCG Loss: Maximize the estimated DCG

    In this step, we maximize the estimated DCG (EDCG) while fixing the surrogate loss model 
    parameters. The loss function is defined as:

    $$
    \mathcal{L}_{\text{DCG}} = -\sum_{u \in \mathcal{U}} \text{EDCG}(u)
    $$

    For the original GuidedRec implementation, please refer to: 
    https://github.com/ahmedrashed-ml/GuidedRec/tree/master/ml100k/Scripts
    
    ## References
    - Rashed, A., Grabocka, J., & Schmidt-Thieme, L. (2021, July).
        A guided learning approach for item recommendation via surrogate loss learning.
        In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval (pp. 605-613).
    """
    def __init__(self, model: IRModel, lr: float = 0.001, weight_decay: float = 0.0, 
        neg_num: int = 9) -> None:
        r"""
        ## Function
        The constructor of GuidedRec optimizer.

        ## Arguments
        model: IRModel
            the model to be optimized
        lr: float
            the learning rate
        weight_decay: float
            the weight decay parameter
        neg_num: int
            the number of negative items for each user

        """
        super(GuidedRecOptimizer, self).__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.neg_num = neg_num
        # Surrogate Loss Model Parameters
        N = neg_num + 1
        self.surr_model = self.SurrogateLossModel(N)
        self.surr_model = self.surr_model.to(self.model.device)
        # optimizers
        self.optimizer_main = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        self.optimizer_surr = torch.optim.Adam(
            self.surr_model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

    class SurrogateLossModel(nn.Module):
        r"""
        ## Class
        The Surrogate Loss Model to estimate the DCG.
        """
        def __init__(self, N: int = 10) -> None:
            r"""
            ## Function
            The constructor of SurrogateLossModel.

            ## Arguments
            N: int
                the number of sampled items for each user
            """
            super(GuidedRecOptimizer.SurrogateLossModel, self).__init__()
            self.N = N
            self.MLP = nn.Sequential(
                nn.Linear(2 * N, 2 * N), nn.Tanh(),
                nn.Linear(2 * N, 2 * N), nn.Tanh(),
            )
            self.NFM_FC1 = nn.Linear(2, 2 * N)
            self.NFM_FC2 = nn.Linear(N, 2 * N)
            self.out_FC1 = nn.Linear(2 * N, 8)
            self.out_FC2 = nn.Linear(8, 1)
            self.act = nn.Tanh()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            r"""
            ## Function
            The forward function of the SurrogateLossModel.

            ## Arguments
            x: torch.Tensor, (B, N, 2)
                the input tensor

            ## Returns
            The output tensor, i.e. EDCG, with shape (B)
            """
            assert x.shape[1] == self.N and x.shape[2] == 2, f"Invalid shape of input tensor: {x.shape}, expected: (B, {self.N}, 2)"
            mlp_x = self.MLP(x.view(-1, 2 * self.N))    # (B, 2 * N)
            z = self.act(self.NFM_FC1(x))               # (B, N, 2 * N)
            z = z @ z.transpose(1, 2)                   # (B, N, N)
            z = torch.mean(z, dim=1)                    # (B, N)
            nfm_x = self.act(self.NFM_FC2(z))           # (B, 2 * N)
            out = mlp_x * nfm_x                         # (B, 2 * N)
            out = self.act(self.out_FC1(out))           # (B, 8)
            out = self.act(self.out_FC2(out))           # (B, 1)
            return out.squeeze(1)                       # (B)

    def cal_loss(self, batch: IRDataBatch) -> torch.Tensor:
        r"""
        ## Function
        Calculate the GuidedRec loss for batch data.
        Note that this function only calculates the main loss, i.e., the BCE loss.

        ## Arguments
        batch: IRDataBatch
            the batch data

        ## Returns
        The loss of the batch data.
        """
        user_emb, item_emb, *addition = self.model.embed(norm=self.model.norm)
        user = user_emb[batch.user]                                     # (B, emb_size)
        pos_item = item_emb[batch.pos_item]                             # (B, emb_size)
        neg_items = item_emb[batch.neg_items]                           # (B, N-1, emb_size)
        items = torch.cat([pos_item.unsqueeze(1), neg_items], dim=1)    # (B, N, emb_size)
        scores = F.cosine_similarity(user.unsqueeze(1), items, dim=2)   # (B, N)
        scores = torch.sigmoid(scores)                                  # (B, N)
        # BCE Loss
        bce_pos = torch.log(scores[:, 0])
        bce_neg = torch.log(1 - scores[:, 1:]).sum(dim=1)
        loss = -torch.mean(bce_pos * self.neg_num + bce_neg)
        loss += self.model.additional_loss(batch, user_emb, item_emb, *addition)
        return loss

    def cal_loss_surrogate(self, batch: IRDataBatch) -> torch.Tensor:
        r"""
        ## Function
        Calculate the surrogate loss for batch data.

        ## Arguments
        batch: IRDataBatch
            the batch data

        ## Returns
        The loss of the batch data.
        """
        user_emb, item_emb, *addition = self.model.embed(norm=self.model.norm)
        user = user_emb[batch.user]                                     # (B, emb_size)
        pos_item = item_emb[batch.pos_item]                             # (B, emb_size)
        neg_items = item_emb[batch.neg_items]                           # (B, N-1, emb_size)
        items = torch.cat([pos_item.unsqueeze(1), neg_items], dim=1)    # (B, N, emb_size)
        scores = F.cosine_similarity(user.unsqueeze(1), items, dim=2)   # (B, N)
        scores = torch.sigmoid(scores)                                  # (B, N)
        # estimated DCG
        labels = torch.zeros_like(scores)                               # (B, N)
        labels[:, 0] = 1
        x = torch.cat([scores.unsqueeze(2), labels.unsqueeze(2)], dim=2) # (B, N, 2)
        edcg = self.surr_model(x)                                       # (B)
        # true DCG
        N = self.neg_num + 1
        sort_idx = torch.argsort(scores, dim=1, descending=True)        # (B, N)
        sort_labels = torch.gather(labels, 1, sort_idx)                 # (B, N)
        dcg = torch.sum(sort_labels / 
            torch.log2(torch.arange(2, N + 2).float().to(sort_labels.device)), dim=1) # (B)
        loss = torch.mean((edcg - dcg) ** 2)
        return loss

    def cal_loss_dcg(self, batch: IRDataBatch) -> torch.Tensor:
        r"""
        ## Function
        Calculate the DCG loss for batch data.

        ## Arguments
        batch: IRDataBatch
            the batch data

        ## Returns
        The loss of the batch data.
        """
        user_emb, item_emb, *addition = self.model.embed(norm=self.model.norm)
        user = user_emb[batch.user]                                     # (B, emb_size)
        pos_item = item_emb[batch.pos_item]                             # (B, emb_size)
        neg_items = item_emb[batch.neg_items]                           # (B, N-1, emb_size)
        items = torch.cat([pos_item.unsqueeze(1), neg_items], dim=1)    # (B, N, emb_size)
        scores = F.cosine_similarity(user.unsqueeze(1), items, dim=2)   # (B, N)
        scores = torch.sigmoid(scores)                                  # (B, N)
        # DCG loss
        labels = torch.zeros_like(scores)                               # (B, N)
        labels[:, 0] = 1
        x = torch.cat([scores.unsqueeze(2), labels.unsqueeze(2)], dim=2) # (B, N, 2)
        edcg = self.surr_model(x)                                       # (B)
        loss = -torch.mean(edcg)
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
        # 1. BCE loss
        self.optimizer_main.zero_grad()
        loss_bce = self.cal_loss(batch)
        loss_bce.backward()
        self.optimizer_main.step()
        # 2. Surrogate loss
        loss_surr = self.cal_loss_surrogate(batch)
        self.optimizer_surr.zero_grad()
        loss_surr.backward()
        self.optimizer_surr.step()
        # 3. DCG loss
        loss_dcg = self.cal_loss_dcg(batch)
        self.optimizer_main.zero_grad()
        loss_dcg.backward()
        self.optimizer_main.step()
        return loss_bce.cpu().item()

    def zero_grad(self) -> None:
        r"""
        ## Function
        Zero the gradients of the optimizer.
        """
        self.optimizer_main.zero_grad()
        self.optimizer_surr.zero_grad()
        

