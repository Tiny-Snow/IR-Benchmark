# -------------------------------------------------------------------
# ItemRec / Item Recommendation Benchmark
# Copyright (C) 2024 Tiny Snow / Weiqin Yang @ Zhejiang University
# -------------------------------------------------------------------
# Module: Experiments Setting Helper
# Description:
#   This module provides an experiment helper to set up experiment
#   settings, so as to ensure reproducibility.
#   This module provides the run function to run the training and
#   testing process for ItemRec.
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
from .logger import logger, set_logfile
from .timer import timer
from ..dataset import *
from ..model import *
from ..optimizer import *
from ..metrics import eval_metrics
import os
from argparse import Namespace
import numpy as np
import random
import torch
import nni
from tqdm import tqdm
import hashlib

# public functions --------------------------------------------------
__all__ = [
    "set_experiments",
    "run",
    "get_info",
]

# global variables --------------------------------------------------
# dataset
dataset: IRDataset = None
# dataloader (training)
dataloader: IRDataLoader = None
# model
model: IRModel = None
# optimizer
optimizer: IROptimizer = None
# device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# random seed --------------------------------------------------------
def set_seed(seed: int) -> None:
    """
    ## Function
    Set random seed for reproducibility.
    Since we ramdomly split the train dataset into train and validation
    sets, you should set up the dataset immediately after setting the
    random seed to ensure that the dataset splits in all experiments
    are the same. 

    ## Arguments
    seed: int
        Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)

# experiment main setting function ---------------------------------
def set_experiments(args: Namespace) -> None:
    """
    ## Function
    Set up experiment settings.
    You should call this function after parsing arguments 
    and before running other functions.

    This function will: 
    - set up logger
    - set up timer
    - set up random seed
    - set up global variables, including model, dataset, dataloader, 
        optimizer, etc.
    - create save directory

    ## Arguments
    args: Namespace
        Arguments from command line interface.
    """
    global dataset, dataloader, model, optimizer, device, additional_trainer
    # set up logger, use info
    info = get_info(args)
    # if info is too long, use hash
    if len(info) > os.pathconf('.', 'PC_NAME_MAX'):
        info = hashlib.md5(info.encode()).hexdigest()
    set_logfile(logger, os.path.join(args.save_dir, f"{info}.log"))
    # log arguments
    logger.info(f"Arguments: {args}.")
    # set up timer
    timer.change_logger(logger)
    # set random seed
    set_seed(args.seed)
    logger.info(f"Random seed has been set to {args.seed}.")
    # set up dataset
    # NOTE: if you have defined your own dataset, you should specify it
    dataset = IRDataset(args.data_path, no_valid=args.no_valid)
    logger.info(f"Dataset has been loaded from {args.data_path}, where " \
        f"train_size={len(dataset.train_interactions)}, " \
        f"valid_size={len(dataset.valid_interactions)}, " \
        f"test_size={len(dataset.test_interactions)}.")
    # set up dataloader
    # NOTE: if you have defined your own dataloader, you should specify it
    neg_num = args.neg_num if hasattr(args, 'neg_num') else 0
    dataloader = IRDataLoader(dataset, batch_size = args.batch_size, 
        shuffle = True, num_workers = args.num_workers, neg_num = neg_num)
    logger.info(f"Built IRDataLoader(batch_size={args.batch_size}, " \
        f"shuffle=True, num_workers={args.num_workers}, neg_num={neg_num}).")
    # set up model
    # NOTE: if you have defined your own model, you should specify it
    if args.model == 'MF':
        model = MFModel(dataset.user_size, dataset.item_size, args.emb_size, norm = args.norm)
        logger.info(f"Built MFModel(user_size={dataset.user_size}, " \
            f"item_size={dataset.item_size}, emb_size={args.emb_size}, norm={args.norm}).")
    elif args.model == 'LightGCN':
        model = LightGCNModel(dataset.user_size, dataset.item_size, args.emb_size, norm = args.norm, 
            num_layers = args.num_layers, edges = dataset.train_interactions)
        logger.info(f"Built LightGCNModel(user_size={dataset.user_size}, " \
            f"item_size={dataset.item_size}, emb_size={args.emb_size}, norm={args.norm}, " \
            f"num_layers={args.num_layers}, edges={len(dataset.train_interactions)}).")
    elif args.model == 'XSimGCL':
        model = XSimGCLModel(dataset.user_size, dataset.item_size, args.emb_size, norm = args.norm, 
            num_layers = args.num_layers, edges = dataset.train_interactions, 
            contrast_weight = args.contrast_weight, contrast_layer=args.contrast_layer, 
            noise_eps = args.noise_eps, InfoNCE_tau = args.InfoNCE_tau)
        logger.info(f"Built XSimGCLModel(user_size={dataset.user_size}, " \
            f"item_size={dataset.item_size}, emb_size={args.emb_size}, norm={args.norm}, " \
            f"num_layers={args.num_layers}, edges={len(dataset.train_interactions)}, " \
            f"contrast_weight={args.contrast_weight}, contrast_layer={args.contrast_layer}, " \
            f"noise_eps={args.noise_eps}, InfoNCE_tau={args.InfoNCE_tau}).")
    elif args.model == 'NCF':
        model = NCFModel(dataset.user_size, dataset.item_size, args.emb_size, norm = args.norm, 
            layers = args.layers)
        logger.info(f"Built NCFModel(user_size={dataset.user_size}, " \
            f"item_size={dataset.item_size}, emb_size={args.emb_size}, norm={args.norm}, " \
            f"layers={args.layers}).")
    elif args.model == 'SimpleX':
        model = SimpleXModel(dataset.user_size, dataset.item_size, args.emb_size, norm = args.norm, 
            history_len = args.history_len, history_weight = args.history_weight, 
            edges = dataset.train_interactions)
        logger.info(f"Built SimpleXModel(user_size={dataset.user_size}, " \
            f"item_size={dataset.item_size}, emb_size={args.emb_size}, norm={args.norm}, " \
            f"history_len={args.history_len}, history_weight={args.history_weight}, " \
            f"edges={len(dataset.train_interactions)}).")
    else:
        raise ValueError(f"Invalid model: {args.model}.")
    model = model.to(device)
    logger.info(f"Model has been moved to {device}.")
    # set up optimizer
    # NOTE: if you have defined your own optimizer, you should specify it
    if args.optim == 'AdvInfoNCE':
        optimizer = AdvInfoNCEOptimizer(model, args.lr, args.weight_decay,
            neg_num=args.neg_num, tau=args.tau, neg_weight=args.neg_weight,
            lr_adv=args.lr_adv, epoch_adv=args.epoch_adv)
        logger.info(f"Built AdvInfoNCEOptimizer(lr={args.lr}, weight_decay={args.weight_decay}, " \
            f"neg_num={args.neg_num}, tau={args.tau}, neg_weight={args.neg_weight}, " \
            f"lr_adv={args.lr_adv}, epoch_adv={args.epoch_adv}).")
    elif args.optim == 'BPR':
        optimizer = BPROptimizer(model, args.lr, args.weight_decay)
        logger.info(f"Built BPROptimizer(lr={args.lr}, weight_decay={args.weight_decay}).")
    elif args.optim == 'BSL':
        optimizer = BSLOptimizer(model, args.lr, args.weight_decay, 
            neg_num=args.neg_num, tau1=args.tau1, tau2=args.tau2)
        logger.info(f"Built BSLOptimizer(lr={args.lr}, weight_decay={args.weight_decay}, " \
            f"neg_num={args.neg_num}, tau1={args.tau1}, tau2={args.tau2}).")
    elif args.optim == 'GuidedRec':
        optimizer = GuidedRecOptimizer(model, args.lr, args.weight_decay, 
            neg_num=args.neg_num)
        logger.info(f"Built GuidedRecOptimizer(lr={args.lr}, weight_decay={args.weight_decay}, " \
            f"neg_num={args.neg_num}).")
    elif args.optim == 'LambdaRank':
        optimizer = LambdaRankOptimizer(model, args.lr, args.weight_decay)
        logger.info(f"Built LambdaRankOptimizer(lr={args.lr}, weight_decay={args.weight_decay}).")
    elif args.optim == 'LambdaLoss':
        optimizer = LambdaLossOptimizer(model, args.lr, args.weight_decay)
        logger.info(f"Built LambdaLossOptimizer(lr={args.lr}, weight_decay={args.weight_decay}).")
    elif args.optim == 'LambdaLossAtK':
        optimizer = LambdaLossAtKOptimizer(model, args.lr, args.weight_decay, 
            K=args.k)
        logger.info(f"Built LambdaLossAtKOptimizer(lr={args.lr}, weight_decay={args.weight_decay}, " \
            f"K={args.k}).")
    elif args.optim == 'LLPAUC':
        optimizer = LLPAUCOptimizer(model, args.lr, args.weight_decay,
            neg_num=args.neg_num, alpha=args.alpha, beta=args.beta)
        logger.info(f"Built LLPAUCOptimizer(lr={args.lr}, weight_decay={args.weight_decay}, " \
            f"neg_num={args.neg_num}, alpha={args.alpha}, beta={args.beta}).")
    elif args.optim == 'PSL':
        optimizer = PSLOptimizer(model, args.lr, args.weight_decay, 
            neg_num=args.neg_num, tau=args.tau, tau_star=args.tau_star, 
            method=args.method, activation=args.activation)
        logger.info(f"Built PSLOptimizer(lr={args.lr}, weight_decay={args.weight_decay}, " \
            f"neg_num={args.neg_num}, tau={args.tau}, tau_star={args.tau_star}, " \
            f"method={args.method}, activation={args.activation}).")
    elif args.optim == 'SLatK':
        optimizer = SLatKOptimizer(model, args.lr, args.weight_decay,
            neg_num=args.neg_num, tau=args.tau, tau_beta=args.tau_beta, K=args.k, 
            epoch_quantile=args.epoch_quantile, train_dict=dataset.train_dict)
        logger.info(f"Built SLatKOptimizer(lr={args.lr}, weight_decay={args.weight_decay}, " \
            f"neg_num={args.neg_num}, tau={args.tau}, tau_beta={args.tau_beta}, " \
            f"K={args.k}, epoch_quantile={args.epoch_quantile}).")
    elif args.optim == 'Softmax':
        optimizer = SoftmaxOptimizer(model, args.lr, args.weight_decay, 
            neg_num=args.neg_num, tau=args.tau)
        logger.info(f"Built SoftmaxOptimizer(lr={args.lr}, weight_decay={args.weight_decay}, " \
            f"neg_num={args.neg_num}, tau={args.tau}).")
    else:
        raise ValueError(f"Invalid optimizer: {args.optim}.")
    # create save directory
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)


# run --------------------------------------------------------------
def run(num_epochs: int, batch_size: int, num_workers: int, save_dir: str, info: str = '') -> None:
    r"""
    ## Function
    Run the training and testing process for ItemRec.
    You should set up the experiment settings and global variables
    before calling this function.

    ## Arguments
    - num_epochs: int
        the number of epochs for training
    - batch_size: int
        the batch size for training
    - num_workers: int
        the number of workers for dataloader (testing only)
    - save_dir: str
        the directory to save the best model
    - info: str (default: '')
        the information to be appended to the model name
    """
    global dataset, dataloader, model, optimizer, device
    # metrics -------------------------------------------------------
    topks = [1, 5, 20, 50]
    best_metrics = {k: {} for k in topks}
    # training and validation ---------------------------------------
    for epoch in range(num_epochs):
        metrics = {k: {} for k in topks}
        # training --------------------------------------------------
        logger.info(f"Epoch {epoch + 1}/{num_epochs} | Training")
        model.train()
        train_loss = 0.0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            batch.to(device)
            loss = optimizer.step(batch, epoch)
            train_loss += loss
            if torch.isnan(torch.tensor(loss)):
                logger.error(f"NaN detected in loss.")
                nni.report_final_result({'default': 0.0})
                return
        logger.info(f"Epoch {epoch + 1}/{num_epochs} | Training Loss: {train_loss / len(dataloader):.5f}")
        # validation at every 5 epochs
        if (epoch + 1) % 5 == 0:
            logger.info(f"Epoch {epoch + 1}/{num_epochs} | Validation")
            for topk in topks:
                metrics[topk] = eval_metrics(model, dataset, topk, 'valid', batch_size, num_workers)
                for k, v in metrics[topk].items():
                    logger.info(f"Epoch {epoch + 1}/{num_epochs} | Validation | Top-{topk} Metrics: {k}: {v:.5f}")
            # save the best model
            if optimizer.__class__.__name__ == 'SLatKOptimizer':
                if metrics[optimizer.K][f'NDCG@{optimizer.K}'] >= best_metrics[optimizer.K].get(f'NDCG@{optimizer.K}', 0.0):
                    best_metrics = metrics
                    logger.info("Saving the best model ...")
                    torch.save(model, save_dir + f"/best_model_{info}.pt")
            elif metrics[20]['NDCG@20'] >= best_metrics[20].get('NDCG@20', 0.0):
                best_metrics = metrics
                logger.info("Saving the best model ...")
                torch.save(model, save_dir + f"/best_model_{info}.pt")
            # NNI report intermediate results
            report_metrics = metrics[20]
            report_metrics['default'] = metrics[20]['NDCG@20']
            if optimizer.__class__.__name__ == 'SLatKOptimizer':
                report_metrics = metrics[optimizer.K]
                report_metrics['default'] = metrics[optimizer.K][f'NDCG@{optimizer.K}']
            nni.report_intermediate_result(report_metrics)
    # testing
    logger.info(f"Final | Testing")
    # load the best model
    logger.info(f"Loading the best model {save_dir}/best_model_{info}.pt ...")
    model = torch.load(save_dir + f"/best_model_{info}.pt")
    metrics = {k: {} for k in topks}
    for topk in topks:
        metrics[topk] = eval_metrics(model, dataset, topk, 'test', batch_size, num_workers)
        for k, v in metrics[topk].items():
            logger.info(f"Final | Testing | Top-{topk} Metrics: {k}: {v:.5f}")
    # NNI report final results
    report_metrics = metrics[20]
    report_metrics['default'] = metrics[20]['NDCG@20']
    nni.report_final_result(report_metrics)

# get info ---------------------------------------------------------
def get_info(args: Namespace) -> str:
    r"""
    ## Function
    Get the information to be appended to the model name.

    ## Arguments
    args: Namespace
        Arguments from command line interface.

    ## Returns
    - info: str
        The information to be appended to the model name.
    """
    info = ""
    # add model info
    # NOTE: you can add more information to the model name
    info += f"{args.model}_emb({args.emb_size})" + ("_norm" if args.norm else "")
    if args.model == 'MF':
        pass
    elif args.model == 'LightGCN':
        info += f"_layer({args.num_layers})"
    elif args.model == 'XSimGCL':
        info += f"_layer({args.num_layers})_contrast_weight({args.contrast_weight})" \
            f"_contrast_layer({args.contrast_layer})_noise_eps({args.noise_eps})_InfoNCE_tau({args.InfoNCE_tau})"
    elif args.model == 'NCF':
        info += f"_layers({args.layers})"
    elif args.model == 'SimpleX':
        info += f"_history_len({args.history_len})_history_weight({args.history_weight})"
    else:
        raise ValueError(f"Invalid model: {args.model}.")
    # add dataset info (currently do not add)
    # add optim info
    # NOTE: you can add more information to the model name
    info += f"_{args.optim}_lr({args.lr})_wd({args.weight_decay})"
    if args.optim == 'AdvInfoNCE':
        info += f"_neg({args.neg_num})_tau({args.tau})_neg_weight({args.neg_weight})" \
            f"_lr_adv({args.lr_adv})_epoch_adv({args.epoch_adv})"
    elif args.optim == 'BPR':
        pass
    elif args.optim == 'BSL':
        info += f"_neg({args.neg_num})_tau1({args.tau1})_tau2({args.tau2})"
    elif args.optim == 'GuidedRec':
        info += f"_neg({args.neg_num})"
    elif args.optim == 'LambdaRank':
        pass
    elif args.optim == 'LambdaLoss':
        pass
    elif args.optim == 'LambdaLossAtK':
        info += f"_K({args.k})"
    elif args.optim == 'LLPAUC':
        info += f"_neg({args.neg_num})_alpha({args.alpha})_beta({args.beta})"
    elif args.optim == 'PSL':
        info += f"_neg({args.neg_num})_method({args.method})_act({args.activation})"
        info += f"_tau({args.tau})_tau_star({args.tau_star})"
    elif args.optim == 'SLatK':
        info += f"_neg({args.neg_num})_tau({args.tau})_tau_beta({args.tau_beta})" \
            f"_K({args.k})_epoch_quantile({args.epoch_quantile})"
    elif args.optim == 'Softmax':
        info += f"_neg({args.neg_num})_tau({args.tau})"
    else:
        raise ValueError(f"Invalid optimizer: {args.optim}.")
    return info

