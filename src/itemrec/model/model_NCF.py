# -------------------------------------------------------------------
# ItemRec / Item Recommendation Benchmark
# Copyright (C) 2024 Tiny Snow / Weiqin Yang @ Zhejiang University
# -------------------------------------------------------------------
# Module: Model - Neural Collaborative Filtering (NCF)
# Description:
#  This module provides the Neural Collaborative Filtering (NCF) model for ItemRec.
#  Reference:
#  - He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T. S. (2017, April). 
#   Neural collaborative filtering. 
#   In Proceedings of the 26th international conference on world wide web (pp. 173-182).
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
    'NCFModel',
]

# NCF ---------------------------------------------------------------
class NCFModel(IRModel):
    r"""
    ## Class
    The Neural Collaborative Filtering (NCF) model for ItemRec.

    We slightly modify the original NCF model by deleting the output layer
    and directly output the embeddings of users and items. In fact, the NCF 
    here is a MLP model with residual connections. One can check that this
    model is equivalent to the original NCF model.
    
    ## Methods
    NCF overrides the following methods:
    - embed:
        Embed all the user and item ids to user and item embeddings.

    ## References
    - He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T. S. (2017, April). 
        Neural collaborative filtering.
        In Proceedings of the 26th international conference on world wide web (pp. 173-182).
    """
    def __init__(self, user_size: int, item_size: int, emb_size: int, norm: bool = True, 
        layers: List[int] = [32, 16, 8, 64]) -> None:
        r"""
        ## Function
        The constructor of NCF model.

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
        - layers: List[int]
            the sizes of hidden layers, the last layer = emb_size
        """
        super(NCFModel, self).__init__(user_size, item_size, emb_size, norm)
        assert layers[-1] == emb_size, 'The last layer size must be equal to emb_size.'
        self.layers = [emb_size] + layers
        self.mlp_user = self._init_mlp(self.layers)
        self.mlp_item = self._init_mlp(self.layers)

    def _init_mlp(self, layers: List[int]) -> nn.ModuleList:
        r"""
        ## Function
        Initialize the MLP layers.

        ## Arguments
        - layers: List[int]
            the sizes of hidden layers

        ## Returns
        - mlp: nn.ModuleList
            the MLP layers
        """
        mlp = nn.ModuleList()
        for i in range(1, len(layers)):
            mlp.append(nn.Linear(layers[i - 1], layers[i]))
            mlp.append(nn.ReLU())
        return mlp

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
        for layer in self.mlp_user:
            user_emb = layer(user_emb)
        for layer in self.mlp_item:
            item_emb = layer(item_emb)
        user_emb = user_emb + self.user_emb.weight
        item_emb = item_emb + self.item_emb.weight
        if norm:
            user_emb = F.normalize(user_emb, p=2, dim=-1)
            item_emb = F.normalize(item_emb, p=2, dim=-1)
        return user_emb, item_emb

