# -------------------------------------------------------------------
# ItemRec / Item Recommendation Benchmark
# Copyright (C) 2024 Tiny Snow / Weiqin Yang @ Zhejiang University
# -------------------------------------------------------------------
# Module: Metrics
# Description:
#  This module provides the metrics for evaluating the performance of
#  recommendation systems. The metrics include:
#  - Precision@K (Precision in Top-K)
#  - Recall@K (Recall in Top-K)
#  - F1@K (F1 Score in Top-K)
#  - HitRatio@K (Hit Ratio in Top-K)
#  - NDCG@K (Normalized Discounted Cumulative Gain in Top-K)
#  - MRR@K (Mean Reciprocal Rank in Top-K)
#  - AUC (Area Under the ROC Curve)
#  NOTE: For any user, you should make sure that the ground truth items
#  are at least 1 in the evaluation set.
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
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from functools import wraps
from ..utils import logger
from ..dataset import IRDataset
from ..model import IRModel

# public functions --------------------------------------------------
__all__ = [
    'eval_metrics',
]

# preserve_torch_seed -----------------------------------------------
def preserve_torch_seed(func: Callable) -> Callable:
    r"""
    ## Decorator
    Preserve the random seed of PyTorch before and after the function.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        torch_state = torch.get_rng_state()
        result = func(*args, **kwargs)
        torch.set_rng_state(torch_state)
        return result
    return wrapper

# eval_metrics - main function --------------------------------------
@preserve_torch_seed
def eval_metrics(model: IRModel, dataset: IRDataset, topk: int, mode: str, 
    batch_size: int, num_workers: int = 0) -> Dict[str, float]:
    r"""
    ## Function
    Evaluate the performance of the recommendation model in the given dataset
    with Top-K metrics. 
    
    Note that `torch.topk` may not be stable, so calling `eval_metrics` multiple
    times may get different results. We have fixed this issue by the decorator
    `preserve_torch_seed`.

    ## Arguments
    - model: IRModel
        the recommendation model
    - dataset: IRDataset
        the dataset for evaluation
    - topk: int
        the K of Top-K metrics, i.e. we evaluate the performance of the model
        in the top-K items recommended by the model
    - mode: str ('valid' or 'test')
        the mode of evaluation, i.e. we evaluate the performance of the model
        on the validation set or the test set
    - batch_size: int
        the batch size for evaluation
    - num_workers: int
        the number of workers for the dataloader

    ## Returns
    - metrics: Dict[str, float]
        the evaluation metrics, including:
        - precision@K
        - recall@K
        - F1@K
        - HitRatio@K
        - NDCG@K
        - MRR@K
        - AUC (very slow, currently not used)
    """
    dataloader = DataLoader(torch.arange(dataset.user_size), batch_size=batch_size, shuffle=False, num_workers=num_workers)
    metrics = {
        f'Precision@{topk}': 0.0,
        f'Recall@{topk}': 0.0,
        f'F1@{topk}': 0.0,
        f'HitRatio@{topk}': 0.0,
        f'NDCG@{topk}': 0.0,
        f'MRR@{topk}': 0.0,
        'AUC': 0.0,
    }
    num_hits, num_test = 0, 0   # used for HitRatio@K
    device = model.device
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f'Evaluating {topk}'):
            batch = batch.to(device)                            # (B)
            # get the scores of all items
            scores = model.scores(batch).cpu()                  # (B, item_size)
            batch = batch.cpu().detach().numpy()
            # mask the positive items in the training set
            mask_items = np.array(dataset.train_dict, dtype=object)[batch]   # (B, ...)
            batch_size, item_size = scores.shape
            for i in range(batch_size):
                scores[i, mask_items[i]] = 0        # scores in [0, 1]
            # get the top-K items
            _, topk_items = torch.topk(scores, topk)            # (B, topk)
            # calculate the metrics
            for i in range(batch_size):
                # get the top-K items and the ground truth items
                topk_item = topk_items[i]
                if mode == 'valid':
                    test_item = set(dataset.valid_dict[batch[i]])
                elif mode == 'test':
                    test_item = set(dataset.test_dict[batch[i]])
                else:
                    raise ValueError(f'Invalid mode for evaluation: {mode}, you should choose from "valid" or "test".')
                # Precision@K
                precision_k = eval_precision(topk_item, test_item)
                metrics[f'Precision@{topk}'] += precision_k
                # Recall@K
                recall_k = eval_recall(topk_item, test_item)
                metrics[f'Recall@{topk}'] += recall_k
                # F1@K
                f1_k = 2 * precision_k * recall_k / (precision_k + recall_k) if precision_k + recall_k > 0 else 0
                metrics[f'F1@{topk}'] += f1_k
                # HitRatio@K
                hit_num = precision_k * topk
                num_hits += hit_num
                num_test += len(test_item)
                # NDCG@K
                ndcg_k = eval_ndcg(topk_item, test_item)
                metrics[f'NDCG@{topk}'] += ndcg_k
                # MRR@K
                mrr_k = eval_mrr(topk_item, test_item)
                metrics[f'MRR@{topk}'] += mrr_k
                # AUC NOTE: very slow !!!
                # auc = eval_auc(scores[i], test_item)
                # metrics['AUC'] += auc
    # average the metrics
    for key in metrics:
        metrics[key] /= dataset.user_size
    metrics[f'HitRatio@{topk}'] = num_hits / num_test
    return metrics
        

# Precision@K -------------------------------------------------------
def eval_precision(topk_items: torch.Tensor, test_items: Set[int]) -> float:
    r"""
    ## Function
    Evaluate the precision of the recommendation model in the given top-K items.

    ## Arguments
    - topk_items: torch.Tensor(K)
        the top-K items recommended by the model
    - test_items: Set[int]
        the ground truth items

    ## Returns
    - precision: float
        the precision of the recommendation model in the given top-K items
    """
    topk_items = set(topk_items.tolist())
    return len(topk_items & test_items) / len(topk_items)

# Recall@K ----------------------------------------------------------
def eval_recall(topk_items: torch.Tensor, test_items: Set[int]) -> float:
    r"""
    ## Function
    Evaluate the recall of the recommendation model in the given top-K items.

    ## Arguments
    - topk_items: torch.Tensor(K)
        the top-K items recommended by the model
    - test_items: Set[int]
        the ground truth items

    ## Returns
    - recall: float
        the recall of the recommendation model in the given top-K items
    """
    topk_items = set(topk_items.tolist())
    return len(topk_items & test_items) / len(test_items) if len(test_items) > 0 else 0.0

# NDCG@K -----------------------------------------------------------
def eval_ndcg(topk_items: torch.Tensor, test_items: Set[int]) -> float:
    r"""
    ## Function
    Evaluate the NDCG (Normalized Discounted Cumulative Gain) of the 
    recommendation model in the given top-K items.

    ## Arguments
    - topk_items: torch.Tensor(K)
        the top-K items recommended by the model
    - test_items: Set[int]
        the ground truth items

    ## Returns
    - ndcg: float
        the NDCG of the recommendation model in the given top-K items
    """
    topk_items = topk_items.tolist()
    K = len(topk_items)
    dcg = 0.0
    for i, item in enumerate(topk_items):
        dcg += 1 / np.log2(i + 2) if item in test_items else 0
    idcg = sum(1 / np.log2(i + 2) for i in range(min(K, len(test_items))))
    return dcg / idcg if idcg > 0 else 0.0

# MRR@K ------------------------------------------------------------
def eval_mrr(topk_items: torch.Tensor, test_items: Set[int]) -> float:
    r"""
    ## Function
    Evaluate the MRR (Mean Reciprocal Rank) of the recommendation model
    in the given top-K items.

    ## Arguments
    - topk_items: torch.Tensor(K)
        the top-K items recommended by the model
    - test_items: Set[int]
        the ground truth items

    ## Returns
    - mrr: float
        the MRR of the recommendation model in the given top-K items
    """
    topk_items = topk_items.tolist()
    mrrs = [1 / (i + 1) for i, item in enumerate(topk_items) if item in test_items]
    return sum(mrrs) / len(mrrs) if len(mrrs) > 0 else 0.0

# AUC --------------------------------------------------------------
def eval_auc(scores: torch.Tensor, test_items: Set[int]) -> float:
    r"""
    ## Function
    Evaluate the AUC (Area Under the ROC Curve) of the recommendation model
    in the given scores and labels.

    ## Arguments
    - scores: torch.Tensor(N)
        the scores of the items
    - test_items: Set[int]
        the ground truth items

    ## Returns
    - auc: float
        the AUC of the recommendation model in the given scores and labels
    """
    labels = np.zeros(len(scores))
    labels[list(test_items)] = 1
    return roc_auc_score(labels, scores)

