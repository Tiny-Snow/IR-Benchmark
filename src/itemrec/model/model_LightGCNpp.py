# -------------------------------------------------------------------
# ItemRec / Item Recommendation Benchmark
# Copyright (C) 2024 Tiny Snow / Weiqin Yang @ Zhejiang University
# -------------------------------------------------------------------
# Module: Model - LightGCN++
# Description:
#  This module provides the LightGCN++ model for item recommendation.
#  Reference:
#  - Lee, Geon, Kyungho Kim, and Kijung Shin. 
#   "Revisiting LightGCN: Unexpected Inflexibility, Inconsistency, and A Remedy Towards Improved Recommendation." 
#   Proceedings of the 18th ACM Conference on Recommender Systems. 2024.
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
    'LightGCNPPModel',
]

# LightGCN++ --------------------------------------------------------
class LightGCNPPModel(IRModel):
    r"""
    ## Class
    The LightGCN++ model for ItemRec.
    
    ## Methods
    LightGCN++ overrides the following methods:
    - embed:
        Embed all the user and item ids to user and item embeddings.

    ## References
    - Lee, Geon, Kyungho Kim, and Kijung Shin.
        "Revisiting LightGCN: Unexpected Inflexibility, Inconsistency, and A Remedy Towards Improved Recommendation."
        Proceedings of the 18th ACM Conference on Recommender Systems. 2024.
    """
    def __init__(self, user_size: int, item_size: int, emb_size: int, norm: bool = True,
        num_layers: int = 3, alpha: float = 0.6, beta: float = -0.1, gamma: float = 0.2,
        edges: List[Tuple[int, int]] = None) -> None:
        r"""
        ## Function
        The constructor of LightGCN++ model.

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
        - num_layers: int
            the number of layers in the LightGCN++ model, default is 3
        - alpha: float
            the norm scaling factor, default is 0.6
        - beta: float
            the neighbor weighting factor, default is -0.1
        - gamma: float
            the weight of the first layer in the final pooling, default is 0.2
        - edges: List[Tuple[int, int]]
            the edges of the graph, i.e. the user-item interactions
        """
        super(LightGCNPPModel, self).__init__(user_size, item_size, emb_size, norm)
        self.num_layers = num_layers
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        # initialize the embeddings
        self._init_weights()
        # build the normalized graph and register it as a buffer
        # sparse matrix, (user_size + item_size, user_size + item_size)
        self._graph = self._build_graph(edges)
        self.register_buffer('graph', self._graph)

    def _init_weights(self):
        r"""
        ## Function
        Initialize the weights of the model.
        LightGCN recommends to use the normal instead of Xavier initialization.
        Note that nn.Embedding just uses normal initialization by default.
        
        NOTE: 
        In our benchmark, we did not detect significant difference between the two 
        initialization methods, which mainly because we use cosine similarity in
        both training and evaluation.
        """
        # nn.init.xavier_normal_(self.user_emb.weight)
        # nn.init.xavier_normal_(self.item_emb.weight)

    def _build_graph(self, edges: List[Tuple[int, int]]) -> torch.Tensor:
        r"""
        ## Function
        Build the normalized graph (COO format sparse matrix) from the user-item interactions.
        Note that LightGCN++ uses a flexible norm scaling and neighbor weighting strategy, i.e.,
        $$
        e_i^{k+1} = \sum_{u \in N_i} 1 / (|N_i|^alpha * |N_u|^beta) * e_u^k / ||e_u^k||_2
        $$
        where $N_i$ is the neighbors of item i and $e_i^k$ is the embedding of item i in layer k.

        ## Arguments
        - edges: List[Tuple[int, int]]
            the edges of the graph, i.e. the user-item interactions

        ## Returns
        - graph: torch.Tensor
            the normalized adjacency matrix {p_{ui}} of the graph, 
            where p_{ui} = 1 / sqrt(deg(u) * deg(i)) and deg(u) is the degree of user u.
        """
        size = self.user_size + self.item_size
        edges = [(u, v + self.user_size) for u, v in edges] + [(v + self.user_size, u) for u, v in edges]
        # get the degree of each node and normalize the edges
        deg = torch.zeros(size)
        for u, v in edges:
            deg[u] += 1
        values = [1 / ((deg[u] ** self.alpha) * (deg[v] ** self.beta)) for u, v in edges]
        # get the sparse matrix
        row, col = zip(*edges)
        graph = torch.sparse_coo_tensor(
            torch.tensor([row, col]), 
            torch.tensor(values),
            size=(size, size)
        ).coalesce()    # coalesce to make the matrix more efficient
        return graph

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
        embs = torch.cat([user_emb, item_emb], dim=0)   # (user_size + item_size, emb_size)
        out_embs = [embs]
        # do Light Graph Convolution (i.e. non-parametric graph convolution)
        # LightGCN++ adds normalization for the embeddings in each layer
        for _ in range(self.num_layers):
            embs = F.normalize(embs, p=2, dim=1)
            embs = torch.sparse.mm(self.graph, embs)
            out_embs.append(embs)
        # mean all the embeddings, the first layer is weighted by gamma
        embs = self.gamma * out_embs[0] + (1 - self.gamma) * torch.stack(out_embs[1:], dim=1).mean(dim=1)
        user_emb = embs[:self.user_size]
        item_emb = embs[self.user_size:]
        if norm:
            user_emb = F.normalize(user_emb, p=2, dim=1)
            item_emb = F.normalize(item_emb, p=2, dim=1)
        return user_emb, item_emb

