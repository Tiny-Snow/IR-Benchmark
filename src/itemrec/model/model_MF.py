# -------------------------------------------------------------------
# ItemRec / Item Recommendation Benchmark
# Copyright (C) 2024 Tiny Snow / Weiqin Yang @ Zhejiang University
# -------------------------------------------------------------------
# Module: Model - Matrix Factorization (MF)
# Description:
#  This module provides the Matrix Factorization (MF) model for ItemRec.
#  Reference:
#  - Y. Koren, R. Bell and C. Volinsky, "Matrix Factorization Techniques for Recommender Systems," 
#   in Computer, vol. 42, no. 8, pp. 30-37, Aug. 2009, doi: 10.1109/MC.2009.263.
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
    'MFModel',
]

# MF ----------------------------------------------------------------
class MFModel(IRModel):
    r"""
    ## Class
    The Matrix Factorization (MF) model for ItemRec.
    
    ## Methods
    MF overrides the following methods:
    - embed:
        Embed all the user and item ids to user and item embeddings.

    ## References
    - Y. Koren, R. Bell and C. Volinsky, "Matrix Factorization Techniques for Recommender Systems,"
        in Computer, vol. 42, no. 8, pp. 30-37, Aug. 2009, doi: 10.1109/MC.2009.263.
    """
    def __init__(self, user_size: int, item_size: int, emb_size: int, norm: bool = True) -> None:
        r"""
        ## Function
        The constructor of MF model.

        ## Arguments
        - user_size: int
            the number of users
        - item_size: int
            the number of items
        - emb_size: int
            the size of embeddings
        - norm: bool
            whether to normalize the embeddings in testing, 
        """
        super(MFModel, self).__init__(user_size, item_size, emb_size, norm)

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
        user_emb = self.user_emb.weight
        item_emb = self.item_emb.weight
        if norm:
            user_emb = F.normalize(user_emb, p=2, dim=1)
            item_emb = F.normalize(item_emb, p=2, dim=1)
        return user_emb, item_emb

