# -------------------------------------------------------------------
# ItemRec / Item Recommendation Benchmark
# Copyright (C) 2024 Tiny Snow / Weiqin Yang @ Zhejiang University
# -------------------------------------------------------------------
# Module: Model - SimpleX
# Description:
#  This module provides the SimpleX model for item recommendation.
#  Reference:
#  - Mao, K., Zhu, J., Wang, J., Dai, Q., Dong, Z., Xiao, X., & He, X. (2021, October). 
#   SimpleX: A simple and strong baseline for collaborative filtering. 
#   In Proceedings of the 30th ACM International Conference on Information & Knowledge Management (pp. 1243-1252).
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
from .model_Base import IRModel

# public functions --------------------------------------------------
__all__ = [
    'SimpleXModel',
]

# SimpleX -----------------------------------------------------------
class SimpleXModel(IRModel):
    r"""
    ## Class
    The SimpleX model for ItemRec.

    In this implementation, we set the aggregation function as `mean`.

    Please refer to the following implementation for more details:
    - https://github.com/RUCAIBox/RecBole/blob/master/recbole/model/general_recommender/simplex.py
    
    ## Methods
    SimpleX overrides the following methods:
    - embed:
        Embed all the user and item ids to user and item embeddings.

    ## References
    - Mao, K., Zhu, J., Wang, J., Dai, Q., Dong, Z., Xiao, X., & He, X. (2021, October).
        SimpleX: A simple and strong baseline for collaborative filtering.
        In Proceedings of the 30th ACM International Conference on Information & Knowledge Management (pp. 1243-1252).
    """
    def __init__(self, user_size: int, item_size: int, emb_size: int, norm: bool = True,
        history_len: int = 50, history_weight: float = 0.5, edges: List[Tuple[int, int]] = None) -> None:
        r"""
        ## Function
        The constructor of SimpleX model.

        ## Arguments
        - user_size: int
            the number of users
        - item_size: int
            the number of items
        - emb_size: int
            the size of embeddings
        - norm: bool
            whether to normalize the embeddings in testing,
            note that the embeddings are always normalized in training.
        - history_len: int
            the maximum number of historical items for each user
        - history_weight: float
            the weight of historical items in the user embedding
        - edges: List[Tuple[int, int]]
            the historical user-item interactions
        """
        super(SimpleXModel, self).__init__(user_size, item_size + 1, emb_size, norm)
        self.history_len = history_len
        self.history_weight = history_weight
        self.history_linear = nn.Linear(emb_size, emb_size)
        self.dropout = nn.Dropout(0.1)
        # pad a zero item for the padding of historical items
        self._pad_item_emb()
        # build the historical user-item interactions and register it as a buffer
        # (user_size, history_len)
        self._history_items = self._build_history_items(edges, history_len)
        self.register_buffer('history_items', self._history_items)

    def _pad_item_emb(self):
        r"""
        ## Function
        Pad a zero item embedding for the padding of historical items.
        """
        self.item_emb = nn.Embedding(self.item_size + 1, self.emb_size, padding_idx=self.item_size)

    def _build_history_items(self, edges: List[Tuple[int, int]], history_len: int) -> torch.Tensor:
        r"""
        ## Function
        Build the historical user-item interactions.

        ## Arguments
        - edges: List[Tuple[int, int]]
            the edges of the graph, i.e. the user-item interactions

        ## Returns
        - history_items: torch.Tensor, shape=(user_size, history_len)
            the historical user-item interactions
        """
        history = [[] for _ in range(self.user_size)]
        for u, i in edges:
            history[u].append(i)
        history_items = torch.full((self.user_size, history_len), self.item_size, dtype=torch.long)
        for u, items in enumerate(history):
            history_items[u, : len(items)] = torch.tensor(items[: history_len], dtype=torch.long)
        return history_items

    def embed(self, norm: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        ## Function
        Embed all the user and item ids to user and item embeddings.

        ## Arguments
        - norm: bool
            whether to normalize the embeddings

        ## Returns
        - user_emb: torch.Tensor
            the user embeddings
        - item_emb: torch.Tensor
            the item embeddings
        """
        user_emb = self.user_emb.weight                 # (user_size, emb_size)
        item_emb = self.item_emb.weight                 # (item_size, emb_size)
        history_emb = self.item_emb(self.history_items) # (user_size, history_len, emb_size)
        # aggregate the historical items and the user embedding
        history_emb = history_emb.mean(dim=1)           # (user_size, emb_size)
        history_emb = self.history_linear(history_emb)  # (user_size, emb_size)
        user_emb = (1 - self.history_weight) * user_emb + self.history_weight * history_emb
        if norm:
            user_emb = F.normalize(user_emb, p=2, dim=-1)
            item_emb = F.normalize(item_emb, p=2, dim=-1)
        return user_emb, item_emb

