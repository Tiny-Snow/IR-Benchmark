# -------------------------------------------------------------------
# ItemRec / Item Recommendation Benchmark
# Copyright (C) 2024 Tiny Snow / Weiqin Yang @ Zhejiang University
# -------------------------------------------------------------------
# Module: Model - Base Model
# Description:
#  This module provides the Base Model for ItemRec -- IRModel.
#  All models should be inherited from BaseModel, the standard and base
#  model class for ItemRec.
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
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from ..dataset import IRDataBatch

# public functions --------------------------------------------------
__all__ = [
    'IRModel',
]

# IRModel -----------------------------------------------------------
class IRModel(nn.Module, ABC):
    r"""
    ## Class
    The standard and abstract base model class for ItemRec

    ## Attributes
    This abstract class already includes the following attributes:
    - user_emb: nn.Embedding
        the user embeddings
    - item_emb: nn.Embedding
        the item embeddings

    ## Methods
    You should at least implement the following methods:
    - embed:
        Embed all the user and item ids to user and item embeddings.

    We also provide the following methods:
    - scores:
        Calculate the similarity scores of all items for the given user ids.
    - device:
        The device of the model.
    - additional_loss:
        Calculate the additional loss of the model.
    """
    def __init__(self, user_size: int, item_size: int, emb_size: int, norm: bool = True) -> None:
        r"""
        ## Function
        Initialize the IRModel object.

        ## Arguments
        user_size: int
            the number of users
        item_size: int
            the number of items
        emb_size: int
            the size of user and item embeddings
        norm: bool
            whether to normalize the embeddings in testing, 
            note that the embeddings are always normalized in training.
        """
        super(IRModel, self).__init__()
        self.user_size = user_size
        self.item_size = item_size
        self.emb_size = emb_size
        self.norm = norm
        self.user_emb = nn.Embedding(user_size, emb_size)
        self.item_emb = nn.Embedding(item_size, emb_size)
    
    @abstractmethod
    def embed(self, norm: bool = True) -> Tuple[torch.Tensor, torch.Tensor, Optional[Any]]:
        r"""
        ## Function
        Embed all the user and item ids to user and item embeddings.
        Return the user and item embeddings. If you need to add additional loss,
        you can return the additional embeddings or other information.

        ## Arguments
        norm: bool
            whether to normalize the embeddings

        ## Returns
        Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
            the user and item embeddings, with shapes (user_size, emb_size) 
            and (item_size, emb_size). If you need to add additional loss,
            you can return the additional embeddings, etc.
        """
        pass
    
    def scores(self, user: torch.Tensor) -> torch.Tensor:
        r"""
        ## Function
        Calculate the similarity scores of all items for the given user.
        This method is mainly used for evaluation.
        We highly recommend to use the cosine similarity as the score function, 
        i.e. set `norm` to True in the `embed` method.

        ## Arguments
        user: torch.Tensor((B), dtype=torch.long)
            the user ids

        ## Returns
        torch.Tensor((B, item_size))
            the similarity scores of all items for the given user
        """
        user_emb, item_emb, *_ = self.embed(norm=self.norm)
        user_emb = user_emb[user]
        similarity = user_emb @ item_emb.t()
        similarity = torch.sigmoid(similarity)  # normalize the scores into the range of [0, 1]
        return similarity

    @property
    def device(self) -> torch.device:
        r"""
        ## Property
        The device of the model.
        """
        return next(self.parameters()).device

    def additional_loss(self, batch: IRDataBatch, user_emb: torch.Tensor, item_emb: torch.Tensor, *args) -> torch.Tensor:
        r"""
        ## Function
        Calculate the additional loss of the model.
        This method is mainly used for the additional loss in the training process, 
        e.g. the contrastive loss in contrastive learning models.

        By default, the additional loss is 0.
        If you need to add additional loss, you should override this method.

        ## Arguments
        - batch: IRDataBatch
            the batch data, with shapes:
            - user: torch.Tensor((B), dtype=torch.long)
                the user ids
            - pos_item: torch.Tensor((B), dtype=torch.long)
                the positive item ids
            - neg_items: torch.Tensor((B, 1), dtype=torch.long)
                the negative item ids
        - user_emb: torch.Tensor
            the user embeddings
        - item_emb: torch.Tensor
            the item embeddings
        - *args
            the additional arguments for the additional loss

        ## Returns
        torch.Tensor
            the additional loss of the model
        """
        return torch.tensor(0.0, device=self.device)

