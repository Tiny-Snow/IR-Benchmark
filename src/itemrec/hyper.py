# -------------------------------------------------------------------
# ItemRec / Item Recommendation Benchmark
# Copyright (C) 2024 Tiny Snow / Weiqin Yang @ Zhejiang University
# -------------------------------------------------------------------
# Module: Hyper Parameters Search Configuration
# Description:
#  This module provides the hyper parameters search configuration
#  for NNI (Neural Network Intelligence).
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
import nni
import argparse

# public functions --------------------------------------------------
__all__ = [
    'get_search_space',
    'get_params',
]

# search space ------------------------------------------------------
search_space_dict = {
    'AdvInfoNCE': {
        'lr': {'_type': 'choice', '_value': [0.1, 0.01, 0.001]},
        'weight_decay': {'_type': 'choice', '_value': [0, 1e-4, 1e-5, 1e-6]},
        'tau': {'_type': 'choice', '_value': [0.01, 0.05, 0.1, 0.2, 0.5]},
        'neg_weight': {'_type': 'choice', '_value': [64]},
        'lr_adv': {'_type': 'choice', '_value': [5e-5]},
        'epoch_adv': {'_type': 'choice', '_value': [5]},
    },
    'BPR': {
        'lr': {'_type': 'choice', '_value': [0.1, 0.01, 0.001, 0.0001]},
        'weight_decay': {'_type': 'choice', '_value': [0, 1e-4, 1e-5, 1e-6]},
    }, 
    'BSL': {
        'lr': {'_type': 'choice', '_value': [0.1, 0.01, 0.001]},
        'weight_decay': {'_type': 'choice', '_value': [0, 1e-4, 1e-5, 1e-6]},
        'tau1': {'_type': 'choice', '_value': [0.01, 0.05, 0.1, 0.2, 0.5]},
        'tau2': {'_type': 'choice', '_value': [0.01, 0.05, 0.1, 0.2, 0.5]},
    },
    'GuidedRec': {
        'lr': {'_type': 'choice', '_value': [0.1, 0.01, 0.001]},
        'weight_decay': {'_type': 'choice', '_value': [0, 1e-4, 1e-5, 1e-6]},
    },
    'LambdaRank': {
        'lr': {'_type': 'choice', '_value': [0.1, 0.01, 0.001, 0.0001]},
        'weight_decay': {'_type': 'choice', '_value': [0, 1e-4, 1e-5, 1e-6]},
    },
    'LambdaLoss': {
        'lr': {'_type': 'choice', '_value': [0.1, 0.01, 0.001, 0.0001]},
        'weight_decay': {'_type': 'choice', '_value': [0, 1e-4, 1e-5, 1e-6]},
    },
    'LambdaLossAtK': {
        'lr': {'_type': 'choice', '_value': [0.1, 0.01, 0.001, 0.0001]},
        'weight_decay': {'_type': 'choice', '_value': [0, 1e-4, 1e-5, 1e-6]},
        'k': {'_type': 'choice', '_value': [5, 20, 50]},
    },
    'LLPAUC': {
        'lr': {'_type': 'choice', '_value': [0.1, 0.01, 0.001]},
        'weight_decay': {'_type': 'choice', '_value': [0, 1e-4, 1e-5, 1e-6]},
        'alpha': {'_type': 'choice', '_value': [0.1, 0.3, 0.5, 0.7, 0.9]},
        'beta': {'_type': 'choice', '_value': [0.01, 0.1]},
    },
    'PSL': {
        'lr': {'_type': 'choice', '_value': [0.1, 0.01, 0.001]},
        'weight_decay': {'_type': 'choice', '_value': [0, 1e-4, 1e-5, 1e-6]},
        'tau_star': {'_type': 'choice', '_value': [0.005, 0.025, 0.05, 0.1, 0.25]},
    },
    'SLatK': {
        'lr': {'_type': 'choice', '_value': [0.1, 0.01, 0.001]},
        'weight_decay': {'_type': 'choice', '_value': [0, 1e-4, 1e-5, 1e-6]},
        'tau': {'_type': 'choice', '_value': [0.01, 0.05, 0.1, 0.2, 0.5]},      # NOTE: using the optimal value of Softmax
        'tau_beta': {'_type': 'quniform', '_value': [0.5, 3, 0.25]},
        'k': {'_type': 'choice', '_value': [5, 20, 50]},
        'lambda_topk': {'_type': 'choice', '_value': [0]},
        'lr_quantile': {'_type': 'choice', '_value': [0.001]},
        'epoch_quantile': {'_type': 'choice', '_value': [5, 20]},
        'init_beta': {'_type': 'choice', '_value': [0.0]},
        'slatk_start_epoch': {'_type': 'choice', '_value': [0]},
    },
    'Softmax': {
        'lr': {'_type': 'choice', '_value': [0.1, 0.01, 0.001]},
        'weight_decay': {'_type': 'choice', '_value': [0, 1e-4, 1e-5, 1e-6]},
        'tau': {'_type': 'choice', '_value': [0.01, 0.05, 0.1, 0.2, 0.5]},
    },
}

# get search space --------------------------------------------------
def get_search_space(optim: str) -> Dict[str, Any]:
    r"""
    ## Function
    Get the search space for hyper parameters search.
    
    ## Arguments
    optim: str
        the name of the optimizer
    
    ## Returns
    Dict[str, Any]
        the search space for hyper parameters search
    """
    return search_space_dict[optim]

# get hyper parameters ---------------------------------------------
def get_params(args: argparse.Namespace) -> argparse.Namespace:
    r"""
    ## Function
    Get hyper parameters for the current experiment.
    If not using NNI, the hyper parameters will remain unchanged.
    
    ## Arguments
    args: argparse.Namespace
        the arguments of the current experiment
    
    ## Returns
    argparse.Namespace
        the hyper parameters for the current experiment
    """
    # get hyper parameters
    params = nni.get_next_parameter()
    # update hyper parameters
    for key, value in params.items():
        setattr(args, key, value)
    return args


