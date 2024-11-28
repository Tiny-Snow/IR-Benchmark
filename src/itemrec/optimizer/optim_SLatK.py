# -------------------------------------------------------------------
# ItemRec / Item Recommendation Benchmark
# Copyright (C) 2024 Tiny Snow / Weiqin Yang @ Zhejiang University
# -------------------------------------------------------------------
# Module: Model - SL@K Optimizer
# Description:
#  This module provides the SL@K (Top-K Softmax Loss) Optimizer for ItemRec.
#  SL@K is a NDCG@K oriented loss function for item recommendation.
#  TODO: update the paper after publication.
#  - Yang, W., Chen, J., Zhang, S., Wu, P., Feng, Y., Chen, C., Wang, C.,
#   Towards Optimizing Top-$K$ Ranking Metrics in Recommender Systems.
#   In Submission.
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
    'SLatKOptimizer',
]

# SLatKOptimizer --------------------------------------------------
class SLatKOptimizer(IROptimizer):
    r"""
    ## Class
    The SL@K Optimizer for ItemRec.
    SL@K is a NDCG@K surrogate loss function for item recommendation.
    The SL@K optimizer is inherited from IROptimizer.

    ## Algorithms

    ### SL@K Loss

    The SL@K loss function is defined as:

    $$
    \mathcal{L}_{\text{SL@}K}(u) = \sum_{i \in \mathcal{P}_u} \sigma_w(s_{ui} - \beta_{u}^{K}) \cdot \log \left( \displaystyle\sum_{j \in \mathcal{I}} \sigma_d(d_{uij}) \right)
    $$

    where 
    - $\mathcal{I}$ is the set of all items;
    - $\mathcal{P}_u$ is the set of positive items for user $u$;
    - $s_{ui}$ is the score of user $u$ on item $i$; 
    - $d_{uij} = f(u, j) - f(u, i)$ for positive item $i$ and negative item $j$; 
    - $\beta_{u}^{K}$ is the score quantile of the top-$K$ positive items for user $u$; 
    - $\sigma_w$ and $\sigma_d$ are the surrogate activations for $\mathbb{I}(\cdot \geq 0)$. 
        Specifically, we set $\sigma_w = sigmoid(x / \tau_w)$ and $\sigma_d = sigmoid(x / \tau_d)$, 
        where $\tau_w$ and $\tau_d$ are the temperature parameters.
    
    ### Quantile Estimation or Regression

    In the implementation, we estimate the Top-$K$ score quantile $\beta_{u}^{K}$ by sorting.
    Specifically, we use the Top-$K$ quantile of all positive items and $N$ sampled negative 
    items to estimate the Top-$K$ score quantile $\beta_{u}^{K}$. 
    
    It's evident that the estimated quantile is biased, i.e., it is less or equal to the true
    quantile. Indeed $\beta_{u}^{K}$ can also be learned by quantile regression with a smaller
    error. However, we find that the sorting method is stable and effective enough in practice.

    ### Optimization

    The SL@K loss can be optimized by a two-stage training process:
    ```
    for epoch in range(epoch_num):
        Fix the Top-$K$ score quantile $\beta(u; K)$, and optimize the model parameters by minimizing the SL@K loss.
        if epoch % epoch_quantile == 0:
            Fix the model parameters, and update the Top-$K$ score quantile $\beta(u; K)$.
    ```
    """
    def __init__(self, model: IRModel, lr: float = 0.1, weight_decay: float = 0.0,
        neg_num: int = 1000, tau: float = 1.0, tau_beta: float = 1.0, K: int = 20,
        epoch_quantile: int = 20, train_dict: List[List[int]] = None) -> None:
        r"""
        ## Function
        The constructor of the SL@K optimizer.

        ## Arguments
        model: IRModel
            the model
        lr: float
            the learning rate for the model parameters
        weight_decay: float
            the weight decay for the model parameters
        neg_num: int
            the number of negative samples
        tau: float
            the temperature parameter for the softmax function
        tau_beta: float
            the temperature parameter for the softmax weights
        K: int
            the Top-$K$ value
        epoch_quantile: int
            the epoch interval for the quantile regression
        train_dict: List[List[int]]
            user -> positive items mapping
        """
        super(SLatKOptimizer, self).__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.neg_num = neg_num
        self.tau = tau
        self.tau_beta = tau_beta
        self.K = K
        self.epoch_quantile = epoch_quantile
        self.init_beta = 0.0
        assert train_dict is not None, 'train_dict, or positive items for each user, is required.'
        self.train_dict, self.mask, self.pos_item_num = self._construct_train_dict(train_dict)
        # model optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        # quantile estimation
        self.beta = torch.full((model.user_size, 1), self.init_beta, dtype=torch.float32, device=model.device)
        # weight sigma function  
        self.weight_sigma = lambda x : torch.sigmoid(x / self.tau_beta)

    def _construct_train_dict(self, train_dict: List[List[int]], cutoff: bool = True) \
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        ## Function
        Construct the positive items dictionary for each user.
        The number of positive items is padded to the same maximum length.

        ## Arguments
        train_dict: List[List[int]]
            the list of positive items for each user
        cutoff: bool
            whether to set the maximum length to the 90-th percentile
        
        ## Returns
        - train_dict: torch.Tensor, shape=(len(train_dict), max(len(train_dict[i])))
            the tensor of positive items for each user
        - mask: torch.Tensor, shape=(len(train_dict), max(len(train_dict[i])))
            the mask for the positive items, 1 for valid, 0 for padding
        - pos_item_num: torch.Tensor, shape=(len(train_dict))
            the number of positive items for each user (not padded nor cutoff)
        """
        pos_item_num = [len(items) for items in train_dict]
        if cutoff:
            max_len = int(np.percentile(pos_item_num, 90))
            train_dict = [items[: max_len] for items in train_dict]
        max_len = max([len(items) for items in train_dict])
        pos_item_num = torch.tensor(pos_item_num, dtype=torch.long, device=self.model.device)
        mask = [[1] * len(items) + [0] * (max_len - len(items)) for items in train_dict]
        mask = torch.tensor(mask, dtype=torch.bool, device=self.model.device)
        train_dict = [items + [0] * (max_len - len(items)) for items in train_dict]
        train_dict = torch.tensor(train_dict, dtype=torch.long, device=self.model.device)
        return train_dict, mask, pos_item_num

    def cal_loss(self, batch: IRDataBatch) -> torch.Tensor:
        r"""
        ## Function
        Calculate the SL@K loss for batch data.

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
            the SL@K loss
        """
        # model embeddings & scores, calculate the softmax loss
        user_emb, item_emb, *addition = self.model.embed(norm=self.model.norm)
        user = user_emb[batch.user]                                 # (B, emb_size)
        pos_item = item_emb[batch.pos_item]                         # (B, emb_size)
        neg_items = item_emb[batch.neg_items]                       # (B, N, emb_size)
        pos_scores = F.cosine_similarity(user, pos_item)            # (B)
        neg_scores = F.cosine_similarity(user.unsqueeze(1), neg_items, dim=2)   # (B, N)
        d = neg_scores - pos_scores.unsqueeze(1)                    # (B, N)
        softmax_loss = torch.logsumexp(d / self.tau, dim=1)         # (B)
        # SL@K weight
        batch_beta = self.beta[batch.user]                          # (B, 1)
        weights = self.weight_sigma(pos_scores - batch_beta.squeeze(1))         # (B)
        # SL@K loss
        loss = (weights * softmax_loss).mean()
        loss += self.model.additional_loss(batch, user_emb, item_emb, *addition)
        return loss

    def cal_quantile(self, batch: IRDataBatch) -> None:
        r"""
        ## Function
        Calculate the quantile (beta) for batch data.

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
            None, but update the beta for each user
        """
        with torch.no_grad():
            # model embeddings & scores
            user_emb, item_emb, *addition = self.model.embed(norm=self.model.norm)
            user = user_emb[batch.user]                                 # (B, emb_size)
            batch_pos_items = self.train_dict[batch.user]               # (B, max_len)
            pos_items = item_emb[batch_pos_items]                       # (B, max_len, emb_size)
            neg_items = item_emb[batch.neg_items]                       # (B, N, emb_size)
            pos_scores = F.cosine_similarity(user.unsqueeze(1), pos_items, dim=2)   # (B, max_len)
            batch_mask = self.mask[batch.user]                          # (B, max_len)
            # pos_scores[~batch_mask] = -1e6
            pos_scores = torch.masked_fill(pos_scores, ~batch_mask, -1e6)
            neg_scores = F.cosine_similarity(user.unsqueeze(1), neg_items, dim=2)   # (B, N)
            # update beta
            scores = torch.cat([pos_scores, neg_scores], dim=1)         # (B, max_len + N)
            beta = torch.topk(scores, self.K, dim=1)[0][:, -1]          # (B)
            self.beta[batch.user] = beta.unsqueeze(1)

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
        # update model
        self.optimizer.zero_grad()
        model_loss = self.cal_loss(batch)
        model_loss.backward()
        self.optimizer.step()
        # update quantile (by sorting)
        if (epoch + 1) % self.epoch_quantile == 0:
            self.cal_quantile(batch)
        return model_loss.cpu().item()

