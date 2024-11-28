# -------------------------------------------------------------------
# ItemRec / Item Recommendation Benchmark
# Copyright (C) 2024 Tiny Snow / Weiqin Yang @ Zhejiang University
# -------------------------------------------------------------------
# Module: Model - XSimGCL
# Description:
#  This module provides the XSimGCL model for item recommendation.
#  Reference:
# - Yu, J., Xia, X., Chen, T., Cui, L., Hung, N. Q. V., & Yin, H. (2023). 
#  XSimGCL: Towards extremely simple graph contrastive learning for recommendation. 
#  IEEE Transactions on Knowledge and Data Engineering.
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
from ..dataset import IRDataBatch

# public functions --------------------------------------------------
__all__ = [
    'XSimGCLModel',
]

# XSimGCL ----------------------------------------------------------------
class XSimGCLModel(IRModel):
    r"""
    ## Class
    The XSimGCL (eXtremely Simple Graph Contrastive Learning) model for ItemRec.
    
    XSimGCL is first a SGL (Self-supervised Graph Learning) model for item 
    recommendation, which is a joint learning framework of two tasks:
    - Graph-based item recommendation: $\mathcal{L}_{\text{rec}}$
    - Graph contrastive learning: $\mathcal{L}_{\text{cl}}$
    
    Thus, the overall objective function is:
    $$
    \mathcal{L} = \mathcal{L}_{\text{rec}} + \lambda \mathcal{L}_{\text{cl}}
    $$
    where $\lambda$ is the trade-off parameter.

    SimGCL is a simple and effective SGL model for item recommendation,
    which discards the graph dropout augmentation and adopts a noise-based
    contrastive learning strategy.

    XSimGCL further simplifies SimGCL by unifying the graph-based item
    recommendation and graph contrastive learning into a single pipeline.
    Note that the noise is only added to the embeddings during training.

    For implementation, XSimGCL is just a $L$-layers LightGCN model with
    noise $\Delta(\mathbf{e}) = sign(\mathbf{e}) \odot \mathcal{N}(0, 1)$
    added to the embedding $\mathbf{e}$, where the noise level is controlled, 
    i.e. $\Vert \Delta(\mathbf{e}) \Vert_2 = \epsilon$. The contrastive 
    learning is then performed on the embeddings with noise between the 
    final averaged embedding and the specificed contrasted layer $l^*$'s
    embedding. The recommendation loss is customized, and the contrastive
    loss is the standard InfoNCE loss with temperature $\tau$. 
    
    By default, we set $L = 3$, $\lambda = 0.2$, $l^* = 1$, $\epsilon = 0.1$, 
    and $\tau = 0.2$.

    You may also refer to the original implementation of XSimGCL at:
    https://github.com/Coder-Yu/SELFRec/blob/main/model/graph/XSimGCL.py

    ## Methods
    XSimGCL overrides the following methods:
    - embed:
        Embed all the user and item ids to user and item embeddings.
    - additional_loss:
        Calculate the contrastive InfoNCE loss between the final embeddings
        and the contrasted embeddings.
    
    ## References
    - Yu, J., Xia, X., Chen, T., Cui, L., Hung, N. Q. V., & Yin, H. (2023).
        XSimGCL: Towards extremely simple graph contrastive learning for recommendation.
        IEEE Transactions on Knowledge and Data Engineering.
    """
    def __init__(self, user_size: int, item_size: int, emb_size: int, norm: bool = True,
        num_layers: int = 3, edges: List[Tuple[int, int]] = None, contrast_weight: float = 0.2, 
        contrast_layer: int = 1, noise_eps: float = 0.1, InfoNCE_tau: float = 0.2):
        r"""
        ## Function
        The constructor of XSimGCL model.

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
            the number of layers in the LightGCN model, default is 3
        - edges: List[Tuple[int, int]]
            the edges of the graph, i.e. the user-item interactions
        - contrast_weight: float
            the weight of the contrastive loss, default is 0.2
        - contrast_layer: int
            the contrasted layer, default is 1
        - noise_eps: float
            the noise level, default is 0.1
        - InfoNCE_tau: float
            the temperature of the softmax in InfoNCE loss, default is 0.2
        """
        super(XSimGCLModel, self).__init__(user_size, item_size, emb_size, norm)
        assert contrast_layer < num_layers, 'The contrasted layer should be less than the number of layers.'
        assert contrast_layer > 0, 'The contrasted layer should be greater than 0.'
        self.num_layers = num_layers
        self.contrast_weight = contrast_weight
        self.contrast_layer = contrast_layer
        self.noise_eps = noise_eps
        self.InfoNCE_tau = InfoNCE_tau
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
        Here we follow the original implementation of XSimGCL.
        However, you should not expect significant difference between
        the normal and Xavier initialization if you know the idea of
        xavier initialization.
        """
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

    def _build_graph(self, edges: List[Tuple[int, int]]) -> torch.Tensor:
        r"""
        ## Function
        Build the normalized graph (COO format sparse matrix) from the user-item interactions.

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
        deg = torch.sqrt(deg)
        values = [1 / (deg[u] * deg[v]) for u, v in edges]
        # get the sparse matrix
        row, col = zip(*edges)
        graph = torch.sparse_coo_tensor(
            torch.tensor([row, col]), 
            torch.tensor(values),
            size=(size, size)
        ).coalesce()    # coalesce to make the matrix more efficient
        return graph

    def embed(self, norm: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        ## Function
        Embed all the user and item ids to user and item embeddings.
        Return the user and item embeddings, including the final embeddings
        and the contrasted embeddings.

        ## Arguments
        - norm: bool
            whether to normalize the embeddings

        ## Returns
        - Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            including the final user embeddings, the final item embeddings,
            the contrasted user embeddings, and the contrasted item embeddings, 
            each with shapes (user_size, emb_size) and (item_size, emb_size), 
            respectively.
        """
        user_emb = self.user_emb.weight
        item_emb = self.item_emb.weight
        embs = torch.cat([user_emb, item_emb], dim=0)   # (user_size + item_size, emb_size)
        out_embs = []   # Note that XSimGCL do not average the first layer's embeddings
        # do LGC with noise
        for _ in range(self.num_layers):
            embs = torch.sparse.mm(self.graph, embs)
            # add noise if training
            if self.training:
                noise = torch.randn_like(embs, device=self.device)
                embs += torch.sign(embs) * F.normalize(noise, p=2, dim=1) * self.noise_eps
            out_embs.append(embs)
        # mean all the embeddings
        embs = torch.stack(out_embs, dim=1).mean(dim=1)
        user_emb = embs[:self.user_size]
        item_emb = embs[self.user_size:]
        cl_embs = out_embs[self.contrast_layer - 1]
        user_cl = cl_embs[:self.user_size]
        item_cl = cl_embs[self.user_size:]
        if norm:
            user_emb = F.normalize(user_emb, p=2, dim=1)
            item_emb = F.normalize(item_emb, p=2, dim=1)
            user_cl = F.normalize(user_cl, p=2, dim=1)
            item_cl = F.normalize(item_cl, p=2, dim=1)
        return user_emb, item_emb, user_cl, item_cl

    @staticmethod
    def _InfoNCE(view1: torch.Tensor, view2: torch.Tensor, tau: float) -> torch.Tensor:
        r"""
        ## Function
        Calculate the contrastive InfoNCE loss between two views.

        ## Arguments
        - view1: torch.Tensor
            the first view, with shape (B, emb_size)
        - view2: torch.Tensor
            the second view, with shape (B, emb_size)
        - tau: float
            the temperature of the softmax

        ## Returns
        - torch.Tensor
            the contrastive loss
        """
        view1 = F.normalize(view1, p=2, dim=1)
        view2 = F.normalize(view2, p=2, dim=1)
        score = view1 @ view2.t() / tau             # (B, B)
        score = F.log_softmax(score, dim=1).diag()  # (B)
        return -score.mean()

    def additional_loss(self, batch: IRDataBatch, user_emb: torch.Tensor, item_emb: torch.Tensor, 
        user_cl: torch.Tensor, item_cl: torch.Tensor) -> torch.Tensor:
        r"""
        ## Function
        Calculate the contrastive InfoNCE loss between the final embeddings 
        and the contrasted embeddings.

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
            the final user embeddings
        - item_emb: torch.Tensor
            the final item embeddings
        - user_cl: torch.Tensor
            the contrasted user embeddings
        - item_cl: torch.Tensor
            the contrasted item embeddings

        ## Returns
        - torch.Tensor
            the contrastive loss
        """
        user = user_emb[batch.user]                             # (B, emb_size)
        item = item_emb[batch.pos_item]                         # (B, emb_size)
        user_cl = user_cl[batch.user]                           # (B, emb_size)
        item_cl = item_cl[batch.pos_item]                       # (B, emb_size)
        user_cl_loss = self._InfoNCE(user, user_cl, self.InfoNCE_tau)
        item_cl_loss = self._InfoNCE(item, item_cl, self.InfoNCE_tau)
        return self.contrast_weight * (user_cl_loss + item_cl_loss)

