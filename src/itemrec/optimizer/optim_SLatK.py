# -------------------------------------------------------------------
# ItemRec / Item Recommendation Benchmark
# Copyright (C) 2024 Tiny Snow / Weiqin Yang @ Zhejiang University
# -------------------------------------------------------------------
# Module: Model - SL@K Optimizer
# Description:
#  This module provides the SL@K (Top-K Softmax Loss) Optimizer for ItemRec.
#  SL@K is a NDCG@K oriented loss function for item recommendation.
#  - TODO: add my paper here.
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
    \mathcal{L}_{\mathrm{SL}@K}(u) = \mathbb{E}_{i \in \mathcal{P}_u} \left[ \frac{\sigma(f_\beta(u, i; K))}{\sum_{i' \in \mathcal{P}_u} \sigma(f_\beta(u, i'; K))} \log \left(\sum_{j \in \mathcal{I}} \sigma(d_{uij})\right) \right] - \lambda \log \sum_{i \in \mathcal{P}_u} \sigma(f_\beta(u, i; K))
    $$

    where 
    - $\mathcal{I}$ is the set of all items ;
    - $\mathcal{P}_u$ is the set of positive items for user $u$ ;
    - $f(u, i)$ is the score of user $u$ on item $i$ ; 
    - $d_{uij} = f(u, j) - f(u, i)$ for positive item $i$ and negative item $j$ ; 
    - $\beta(u; K)$ is the score threshold of the top-$K$ positive items for user $u$ ; 
    - $f_\beta(u, i; K) = f(u, i) - \beta(u; K)$ ;
    - $\lambda$ is the weighting hyper-parameter for the penalty term ;
    - $\sigma$ is a surrogate activation for $\mathbb{I}(\cdot \geq 0)$, e.g. 
        $\sigma(x) = \exp(x / \tau)$, where $\tau$ is the temperature parameter.

    In SL@K loss, $w(u, i; K) := \frac{\sigma(f_\beta(u, i; K))}{\sum_{i' \in \mathcal{P}_u} \sigma(f_\beta(u, i'; K))}$ 
    can be viewed as the weight of the original softmax loss for positive item $i$.
    During training, we will sample all the positive items $i$ to make sure that 
    $\sum_i w(u, i; K) = 1$.

    In SL@K loss, $R(u; K) := -\log \sum_{i \in \mathcal{P}_u} \sigma(f_\beta(u, i; K))$
    can be viewed as the penalty term, which is a surrogate for the Top-$K$ hits, forcing 
    the model to increase the number of top-$K$ positive items. During training, the penalty
    or regularization term $R(u; K)$ will be calculated for each sample $(u, i)$ in the batch.

    Therefore, SL@K loss can be simplified as:

    $$
    \mathcal{L}_{\mathrm{SL}@K}(u) = \mathbb{E}_{i \in \mathcal{P}_u} \left[ w(u, i; K) \cdot \log \left(\sum_{j \in \mathcal{I}} \sigma(d_{uij})\right) \right] + \lambda R(u; K)
    $$

    In practice, we omit the denominator in the weight $w(u, i; K)$, and at the same the
    penalty term $R(u; K)$ is also omitted. Therefore, the SL@K loss can be calculated as:

    $$
    \mathcal{L}_{\mathrm{SL}@K}(u) = \mathbb{E}_{i \in \mathcal{P}_u} \left[ \sigma(f_\beta(u, i; K)) \log \left(\sum_{j \in \mathcal{I}} \sigma(d_{uij})\right) \right]
    $$
    
    ### Quantile Regression

    The Top-$K$ score threshold $\beta(u; K)$ can be learned by quantile regression.
    Specifically, we can define the quantile regression loss as:

    $$
    \mathcal{L}_{\mathrm{quantile}}(u, i, \beta; t) = (1 - t)(f(u, i) - \beta(u; K))_{+} + t(\beta(u; K) - f(u, i))_{+}
    $$

    where $t = K / |\mathcal{I}$ is the Top-$K$ quantile, and $(x)_{+} = \max(x, 0)$ is
    the ReLU function.

    In practice, solving the quantile by sorting is also feasible, with the complexity of
    $O(N \log K)$. We can estimate the Top-$K$ quantile by sorting all the positive items
    and the sampled negative items ($\mathcal{P}_u + |\mathcal{N}| \gg K$).

    ### Optimization

    The SL@K loss can be optimized by a two-stage training process:
    1. Fix the Top-$K$ score threshold $\beta(u; K)$, and optimize the model parameters
        by minimizing the SL@K loss.
    2. Fix the model parameters, and optimize the Top-$K$ score threshold $\beta(u; K)$
        by minimizing the quantile regression loss, or by sorting.
    """
    def __init__(self, model: IRModel, lr: float = 0.1, weight_decay: float = 0.0,
        neg_num: int = 1000, tau: float = 1.0, tau_beta: float = 1.0, K: int = 20,
        lambda_topk: float = 1.0, lr_quantile: float = 0.001, epoch_quantile: int = 20,
        init_beta: float = 0.5, slatk_start_epoch: int = 0, weight_sigma: str = 'sigmoid',
        alternative: bool = False, train_dict: List[List[int]] = None) -> None:
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
        lambda_topk: float
            the weighting hyper-parameter for the penalty term
        lr_quantile: float
            the learning rate for the quantile regression
        epoch_quantile: int
            the epoch interval for the quantile regression
        init_beta: float
            the initial value for the Top-$K$ score threshold
        slatk_start_epoch: int
            the epoch to start the SL@K loss optimization
        weight_sigma: str
            the surrogate activation function for the Heaviside step function
        alternative: bool
            whether to use the alternative optimization strategy (SL and SL@K)
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
        self.lambda_topk = lambda_topk
        self.lr_quantile = lr_quantile
        self.epoch_quantile = epoch_quantile
        self.init_beta = init_beta
        self.slatk_start_epoch = slatk_start_epoch
        assert train_dict is not None, 'train_dict, or positive items for each user, is required.'
        self.alternative = alternative
        self.train_dict, self.mask, self.pos_item_num = self._construct_train_dict(train_dict)
        # model optimizer
        self.optimizer_model = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        # quantile regression
        self.beta = torch.full((model.user_size, 1), init_beta, \
            dtype=torch.float32, device=model.device, requires_grad=True)
        # NOTE: check if the beta is required to be updated by training
        self.beta.requires_grad = False
        self.optimizer_quantile = torch.optim.Adam(
            [{'params': self.beta}],
            lr=self.lr_quantile,
        )
        # weight sigma function
        self.weight_sigma_func = weight_sigma
        if weight_sigma == 'exp':           # NOTE: in exp, the beta will be eliminated
            self.weight_sigma = lambda x : torch.exp(x / self.tau_beta)
        elif weight_sigma == 'relu':        # NOTE: in relu, the truncated score f - beta may be negative
            self.weight_sigma = lambda x : torch.pow(F.relu(x + 1), 1 / self.tau_beta)
        elif weight_sigma == 'sigmoid':     
            self.weight_sigma = lambda x : torch.sigmoid(x / self.tau_beta)
        else:
            raise ValueError(f'Invalid weight_sigma: {weight_sigma}')

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

    def update_quantile(self, epoch: int) -> bool:
        r"""
        ## Function
        Whether to update the Top-$K$ score threshold in this epoch.

        ## Arguments
        epoch: int
            the current epoch

        ## Returns
        flag: bool
            whether to update the Top-$K$ score threshold
        """
        return epoch >= self.slatk_start_epoch and (epoch + 1) % self.epoch_quantile == 0
    
    def update_model_slatk(self, epoch: int) -> bool:
        r"""
        ## Function
        Whether to update the model parameters by SL@K loss in this epoch.
        If not, use the Softmax loss.

        If alternative is False:
            Always use the SL@K loss.
        
        If alternative is True:
            For epoch >= slatk_start_epoch, optimize the model using SL@K loss 
            by 5 epochs and then using the Softmax loss by 5 epochs, alternatively.

        ## Arguments
        epoch: int
            the current epoch

        ## Returns
        flag: bool
            whether to update the model parameters by SL@K loss
        """
        if epoch < self.slatk_start_epoch:
            return False
        if epoch == self.slatk_start_epoch:     # SL -> SL@K
            self.optimizer_model = torch.optim.Adam(
                self.model.parameters(),
                lr=self.lr * 0.1,       # a smaller learning rate is better for SL@K
                weight_decay=self.weight_decay
            )
        if not self.alternative:
            return True
        epoch = epoch - self.slatk_start_epoch
        if epoch % 10 == 0:     # for SL@K
            self.optimizer_model = torch.optim.Adam(
                self.model.parameters(),
                lr=self.lr * 0.1,       # a smaller learning rate is better for SL@K
                weight_decay=self.weight_decay
            )
        elif epoch % 10 == 5:   # for Softmax
            self.optimizer_model = torch.optim.Adam(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay
            )
        return epoch % 10 < 5

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
        return self.cal_slatk_loss(batch)

    def cal_sl_loss(self, batch: IRDataBatch) -> torch.Tensor:
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

    # NOTE: SL@K loss (normalized weight version)
    def cal_slatk_loss_normalized(self, batch: IRDataBatch) -> torch.Tensor:
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
        batch_pos_items = self.train_dict[batch.user]               # (B, max_len)
        batch_mask = self.mask[batch.user]                          # (B, max_len)
        batch_pos_item_emb = item_emb[batch_pos_items]              # (B, max_len, emb_size)
        batch_pos_scores = F.cosine_similarity(user.unsqueeze(1), batch_pos_item_emb, dim=2)    # (B, max_len)
        batch_pos_scores = batch_pos_scores - batch_beta            # (B, max_len)
        if self.weight_sigma_func == 'exp':
            # batch_pos_scores[~batch_mask] = -1e6                    # (B, max_len)
            batch_pos_scores = torch.masked_fill(batch_pos_scores, ~batch_mask, -1e6)
        elif self.weight_sigma_func == 'relu':
            # batch_pos_scores[~batch_mask] = -1                      # (B, max_len)
            batch_pos_scores = torch.masked_fill(batch_pos_scores, ~batch_mask, -1)
        elif self.weight_sigma_func == 'sigmoid':
            # batch_pos_scores[~batch_mask] = -1e6                    # (B, max_len)
            batch_pos_scores = torch.masked_fill(batch_pos_scores, ~batch_mask, -1e6)
        # sum weight
        # batch_pos_scores = self.weight_sigma(batch_pos_scores).mean(dim=1)   # (B)
        # batch_pos_item_num = self.pos_item_num[batch.user]          # (B)
        # batch_pos_scores = batch_pos_scores * batch_pos_item_num    # (B), correct the sum
        # mean weight
        batch_pos_scores = self.weight_sigma(batch_pos_scores).sum(dim=1)   # (B)
        batch_pos_item_num = self.pos_item_num[batch.user]          # (B)
        batch_pos_scores = batch_pos_scores / batch_pos_item_num    # (B), correct the mean
        # weights = self.weight_sigma(pos_scores - batch_beta.squeeze(1)) / batch_pos_scores # (B)
        weights = self.weight_sigma(pos_scores - batch_beta.squeeze(1))
        # penalty term
        penalty = torch.log(batch_pos_scores)                       # (B)
        # SL@K loss
        loss = (weights * softmax_loss).mean() - self.lambda_topk * penalty.mean()
        loss += self.model.additional_loss(batch, user_emb, item_emb, *addition)
        return loss
    
    # NOTE: SL@K loss (not-normalized weight version)
    def cal_slatk_loss(self, batch: IRDataBatch) -> torch.Tensor:
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

    # NOTE: quantile loss (single positive item version)
    def cal_quantile_loss(self, batch: IRDataBatch) -> torch.Tensor:
        r"""
        ## Function
        Calculate the quantile regression loss for batch data.

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
            the quantile regression loss
        """
        # model embeddings & scores
        user_emb, item_emb, *addition = self.model.embed(norm=self.model.norm)
        user = user_emb[batch.user]                                 # (B, emb_size)
        pos_item = item_emb[batch.pos_item]                         # (B, emb_size)
        neg_items = item_emb[batch.neg_items]                       # (B, N, emb_size)
        # quantile regression loss
        pos_scores = F.cosine_similarity(user, pos_item)            # (B)
        neg_scores = F.cosine_similarity(user.unsqueeze(1), neg_items, dim=2)   # (B, N)
        scores = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)   # (B, N + 1)
        batch_beta = self.beta[batch.user]                          # (B, 1)
        scores = scores - batch_beta                                # (B, N + 1)
        t = self.K / (self.neg_num + self.K)                        # NOTE: t based on negative sampling
        loss = (1 - t) * F.relu(scores).mean() + t * F.relu(-scores).mean()
        return loss

    # NOTE: quantile loss (full positive items version)
    def cal_quantile_loss_full(self, batch: IRDataBatch) -> torch.Tensor:
        r"""
        ## Function
        Calculate the quantile regression loss for batch data.

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
            the quantile regression loss
        """
        # model embeddings & scores
        user_emb, item_emb, *addition = self.model.embed(norm=self.model.norm)
        user = user_emb[batch.user]                                 # (B, emb_size)
        batch_pos_items = self.train_dict[batch.user]               # (B, max_len)
        pos_items = item_emb[batch_pos_items]                       # (B, max_len, emb_size)
        neg_items = item_emb[batch.neg_items]                       # (B, N, emb_size)
        # quantile regression loss
        pos_size, neg_size = pos_items.size(1), neg_items.size(1)
        t = self.K / (pos_size + neg_size)      # NOTE: t based on negative sampling
        batch_beta = self.beta[batch.user]                          # (B, 1)
        pos_scores = F.cosine_similarity(user.unsqueeze(1), pos_items, dim=2)   # (B, max_len)
        pos_scores = pos_scores - batch_beta                        # (B, max_len)
        batch_mask = self.mask[batch.user]                          # (B, max_len)
        pos_scores_mask = pos_scores
        pos_scores_mask[~batch_mask] = 0
        pos_loss = (1 - t) * F.relu(pos_scores_mask).mean() + t * F.relu(-pos_scores_mask).mean()
        neg_scores = F.cosine_similarity(user.unsqueeze(1), neg_items, dim=2)   # (B, N)
        neg_scores = neg_scores - batch_beta                        # (B, N)
        neg_loss = (1 - t) * F.relu(neg_scores).mean() + t * F.relu(-neg_scores).mean()
        loss = pos_loss + neg_loss
        return loss

    # NOTE: directly calculate the quantile by sorting
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

    # NOTE: SL/SL@K alternative optimization
    def step_alternative(self, batch: IRDataBatch, epoch: int) -> float:
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
        if not self.update_quantile(epoch):     # update model
            for param in self.model.parameters():
                param.requires_grad = True
            self.beta.requires_grad = False
            self.optimizer_model.zero_grad()
            if self.update_model_slatk(epoch):
                loss = self.cal_slatk_loss(batch)
            else:
                loss = self.cal_sl_loss(batch)
            loss.backward()
            self.optimizer_model.step()
        else:                                   # update quantile
            for param in self.model.parameters():
                param.requires_grad = False
            self.beta.requires_grad = True
            self.optimizer_quantile.zero_grad()
            loss = self.cal_quantile_loss(batch)
            loss.backward()
            self.optimizer_quantile.step()
        return loss.item()

    # NOTE: SL@K quantile regression optimization
    def step_regression(self, batch: IRDataBatch, epoch: int) -> float:
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
        for param in self.model.parameters():
            param.requires_grad = True
        self.beta.requires_grad = False
        self.optimizer_model.zero_grad()
        if self.update_model_slatk(epoch):
            model_loss = self.cal_slatk_loss(batch)
        else:
            model_loss = self.cal_sl_loss(batch)
        model_loss.backward()
        self.optimizer_model.step()
        # update quantile (by quantile loss)
        for param in self.model.parameters():
            param.requires_grad = False
        self.beta.requires_grad = True
        if self.update_quantile(epoch):
            self.optimizer_quantile.zero_grad()
            quantile_loss = self.cal_quantile_loss(batch)
            quantile_loss.backward()
            self.optimizer_quantile.step()
        return model_loss.item()
    
    # NOTE: SL@K quantile sorting optimization
    def step_sorting(self, batch: IRDataBatch, epoch: int) -> float:
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
        self.optimizer_model.zero_grad()
        if self.update_model_slatk(epoch):
            model_loss = self.cal_slatk_loss(batch)
        else:
            model_loss = self.cal_sl_loss(batch)
        model_loss.backward()
        self.optimizer_model.step()
        # update quantile (by sorting)
        if self.update_quantile(epoch):
            self.cal_quantile(batch)
        return model_loss.item()

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
        # SL/SL@K alternative optimization
        # return self.step_alternative(batch, epoch)
        # SL@K quantile regression optimization
        # return self.step_regression(batch, epoch)
        # SL@K quantile sorting optimization
        return self.step_sorting(batch, epoch)

    def zero_grad(self) -> None:
        r"""
        ## Function
        Zero the gradients of the optimizer.
        """
        self.optimizer_model.zero_grad()
        self.optimizer_quantile.zero_grad()

    def beta_info(self) -> str:
        r"""
        ## Function
        Get the information of the Top-$K$ score threshold $\beta$.
        
        ## Returns
        info: str
            the information of $\beta$
        """
        return f'Beta: mean: {self.beta.mean().item():.4f}, std: {self.beta.std().item():.4f}, '\
            f'range: [{self.beta.min().item():.4f}, {self.beta.max().item():.4f}]'
