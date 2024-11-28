# -------------------------------------------------------------------
# ItemRec / Item Recommendation Benchmark
# Copyright (C) 2024 Tiny Snow / Weiqin Yang @ Zhejiang University
# -------------------------------------------------------------------
# Module: Arguments Parser
# Description:
#  This module provides the arguments parser for ItemRec -- args.
#  All arguments should be defined in this module.
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
from . import __version__
import argparse

# public functions --------------------------------------------------
__all__ = [
    'parse_args'
]

# args parser -------------------------------------------------------
def parse_args() -> argparse.Namespace:
    r"""
    ## Function
    Parse the arguments from the command line.

    ## Args Parser
    The overall command is defined as the following:
    ```
    itemrec [-h] [-v] --log LOG --save_dir SAVE_DIR --seed SEED 
    model [--model_args ...] dataset [--dataset_args ...] optim [--optim_args ...]
    ```
    where
    - `--log` (str, optional, default="ir.log")
        The log file for ItemRec.
        NOTE: Currently, the log file is not used.
        We save the log file in the {save_dir}/{info}.log, 
        where {info} is the information of the hyper parameters.
    - `--save_dir` (str, optional, default="ir_save")
        The save folder for ItemRec.
    - `--seed` (int, optional, default=2024)
        The random seed for ItemRec.
    - `dataset` (sub-command)
        The dataset sub-command for ItemRec.
    - `model` (sub-command)
        The model sub-command for ItemRec.
    - `optim` (sub-command)
        The optim sub-command for ItemRec.

    ## Returns
    - args: argparse.Namespace
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        prog="itemrec",
        description="ItemRec: Benckmark datasets, SOTA models and unified CLI for Item Recommendation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-v", "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    # log
    parser.add_argument(
        "--log",
        type=str,
        default="ir.log",
        help="The log file for ItemRec. Currently not used, the log file is saved in the {save_dir}/{info}.log.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="ir_save",
        help="The save folder for ItemRec.",
    )
    # seed
    parser.add_argument(
        "--seed",
        type=int,
        default=2024,
        help="The random seed for ItemRec.",
    )
    # model sub-parser
    model_parser = setup_model_parser()
    # dataset sub-parser
    dataset_parser = setup_dataset_parser()
    # optim sub-parser
    optim_parser = setup_optim_parser()
    subparsers = {
        "model": model_parser,
        "dataset": dataset_parser,
        "optim": optim_parser,
    }
    # parse arguments
    args, args_remain = parser.parse_known_args()
    for subcmd, subparser in subparsers.items():
        if args and args_remain[0] == subcmd:
            sub_args, args_remain = subparser.parse_known_args(args_remain[1:])
            args = merge_args(args, sub_args)
    # check arguments validity
    check_args(args)
    return args

# model sub-parser --------------------------------------------------
def setup_model_parser() -> argparse.ArgumentParser:
    r"""
    ## Function
    Setup the model sub-parser for ItemRec.

    ## Args Parser
    The overall model sub-command is defined as the following:
    ```
    itemrec ... model --emb_size EMB_SIZE [--norm] --num_epochs NUM_EPOCHS
    MODEL_NAME [--model_args ...] ...
    ```
    where
    - `--emb_size` (int, optional, default=64)
        The size of user and item embeddings.
    - `--norm` (bool, optional, default=False)
        Whether to normalize the embeddings (at the final step of embedding).
        If true, the cosine similarity is used in the evaluation, i.e. the embeddings 
        are normalized, else the dot product is used in the evaluation.
        NOTE: The embeddings are always normalized in training.
    - `--num_epochs` (int, optional, default=100)
        The number of epochs for training the model.
    - `MODEL_NAME` (str, required)
        The name of the model, including `MF`, `LightGCN`, `NGCF`, etc.
    
    ### MF
    ```
    itemrec ... model ... MF ...
    ```
    No additional arguments required.

    ### LightGCN
    ```
    itemrec ... model ... LightGCN --num_layers NUM_LAYERS ...
    ```
    where
    - `--num_layers` (int, optional, default=3)
        The number of layers in the LightGCN model.

    ### XSimGCL
    ```
    itemrec ... model ... XSimGCL --num_layers NUM_LAYERS --contrast_weight CONTRAST_WEIGHT
    --contrast_layer CONTRAST_LAYER --noise_eps NOISE_EPS --InfoNCE_tau InfoNCE_tau ...
    ```
    where
    - `--num_layers` (int, optional, default=3)
        The number of layers in the XSimGCL model.
    - `--contrast_weight` (float, optional, default=0.2)
        The weight of the contrastive loss.
    - `--contrast_layer` (int, optional, default=1)
        The layer for the contrastive loss.
    - `--noise_eps` (float, optional, default=0.1)
        The noise epsilon for the contrastive loss.
    - `--InfoNCE_tau` (float, optional, default=0.1)
        The temperature for the InfoNCE loss.
    
    ### NCF
    ```
    itemrec ... model ... NCF --layers LAYERS ...
    ```
    where
    - `--layers` (List[int], optional, default=[32, 16, 8, 64])
        The sizes of hidden layers, the last layer = emb_size.

    ### SimpleX
    ```
    itemrec ... model ... SimpleX --history_len HISTORY_LEN --history_weight HISTORY_WEIGHT ...
    ```
    where
    - `--history_len` (int, optional, default=50)
        The maximum number of historical items for each user.
    - `--history_weight` (float, optional, default=0.5)
        The weight of historical items in the user embedding.

    ## Returns
    - parser: argparse.ArgumentParser
        The sub/stage parser for ItemRec model.
    """
    parser = argparse.ArgumentParser(
        description="The model sub-command for ItemRec.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--emb_size",
        type=int,
        default=64,
        help="The size of user and item embeddings.",
    )
    parser.add_argument(
        "--norm",
        action="store_true",
        help="Whether to normalize the embeddings. If true, the cosine similarity is used in the evaluation.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100,
        help="The number of epochs for training the model.",
    )
    model_subparsers = parser.add_subparsers(
        title="model sub-commands",
        description="The sub-commands for the model sub-command.",
        dest="model",
    )
    # model: MF
    mf_parser = model_subparsers.add_parser(
        "MF",
        help="The MF model for ItemRec.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # model: LightGCN
    lightgcn_parser = model_subparsers.add_parser(
        "LightGCN",
        help="The LightGCN model for ItemRec.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    lightgcn_parser.add_argument(
        "--num_layers",
        type=int,
        default=3,
        help="The number of layers in the LightGCN model.",
    )
    # model: XSimGCL
    xsimgcl_parser = model_subparsers.add_parser(
        "XSimGCL",
        help="The XSimGCL model for ItemRec.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    xsimgcl_parser.add_argument(
        "--num_layers",
        type=int,
        default=3,
        help="The number of layers in the XSimGCL model.",
    )
    xsimgcl_parser.add_argument(
        "--contrast_weight",
        type=float,
        default=0.2,
        help="The weight of the contrastive loss.",
    )
    xsimgcl_parser.add_argument(
        "--contrast_layer",
        type=int,
        default=1,
        help="The layer for the contrastive loss.",
    )
    xsimgcl_parser.add_argument(
        "--noise_eps",
        type=float,
        default=0.1,
        help="The noise epsilon for the contrastive loss.",
    )
    xsimgcl_parser.add_argument(
        "--InfoNCE_tau",
        type=float,
        default=0.1,
        help="The temperature for the InfoNCE loss.",
    )
    # model: NCF
    ncf_parser = model_subparsers.add_parser(
        "NCF",
        help="The NCF model for ItemRec.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ncf_parser.add_argument(
        "--layers",
        type=str,
        default="[32, 16, 8, 64]",
        help="The sizes of hidden layers, the last layer = emb_size.",
    )
    # model: SimpleX
    simplex_parser = model_subparsers.add_parser(
        "SimpleX",
        help="The SimpleX model for ItemRec.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    simplex_parser.add_argument(
        "--history_len",
        type=int,
        default=50,
        help="The maximum number of historical items for each user.",
    )
    simplex_parser.add_argument(
        "--history_weight",
        type=float,
        default=0.5,
        help="The weight of historical items in the user embedding.",
    )
    return parser

# dataset sub-parser ------------------------------------------------
def setup_dataset_parser() -> argparse.ArgumentParser:
    r"""
    ## Function
    Setup the dataset sub-parser for ItemRec.

    ## Args Parser
    The overall dataset sub-command is defined as the following:
    ```
    itemrec ... dataset --data_path DATA_PATH --batch_size BATCH_SIZE --num_workers NUM_WORKERS [--no_valid] ...
    ```
    where
    - `--data_path` (str, required)
        The path to the dataset.
    - `--batch_size` (int, optional, default=1024)
        The batch size for the dataset, folder includes `train.tsv` and `test.tsv`.
    - `--num_workers` (int, optional, default=0)
        The number of workers for the dataset. 
        If by default 0, the dataset will be loaded in the main process.
    - `--no_valid` (bool, optional, default=False)
        Whether to use the validation set.
        If true, the validation set will not be used, and the test set will be used
        for validation.

    ## Args
    - parser: argparse.ArgumentParser
        The sub/stage parser for ItemRec dataset.
    """
    parser = argparse.ArgumentParser(
        description="The dataset sub-command for ItemRec.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="The path to the dataset.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="The batch size for the dataset.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="The number of workers for the dataset.",
    )
    parser.add_argument(
        "--no_valid",
        action="store_true",
        help="Whether to use the validation set.",
    )
    return parser

# optim sub-parser --------------------------------------------------
def setup_optim_parser() -> argparse.ArgumentParser:
    r"""
    ## Function
    Setup the optim sub-parser for ItemRec.

    ## Args Parser
    The overall optim sub-command is defined as the following:
    ```
    itemrec ... optim --lr LR --weight_decay WEIGHT_DECAY OPTIM_NAME [--optim_args ...] ...
    ```
    where
    - `--lr` (float, optional, default=0.001)
        The learning rate for the optimizer.
    - `--weight_decay` (float, optional, default=0.0)
        The weight decay for the optimizer.
    - `OPTIM_NAME` (str, required)
        The name of the optimizer, including `BPR`, `Softmax`, 'PSL', etc.
    
    ### AdvInfoNCE
    ```
    itemrec ... optim ... AdvInfoNCE --neg_num NEG_NUM --tau TAU --neg_weight NEG_WEIGHT --lr_adv LR_ADV --epoch_adv EPOCH_ADV ...
    ```
    where
    - `--neg_num` (int, optional, default=1000)
        The number of negative samples.
    - `--tau` (float, optional, default=1.0)
        The temperature parameter.
    - `--neg_weight` (float, optional, default=64)
        The negative weight.
    - `--lr_adv` (float, optional, default=0.0001)
        The learning rate for adversarial learning.
    - `--epoch_adv` (int, optional, default=1)
        The epoch interval for adversarial learning.

    ### BPR
    ```
    itemrec ... optim ... BPR ...
    ```
    No additional arguments required.
    The `--neg_num` is set to constant 1 and should not be changed.

    ### BSL
    ```
    itemrec ... optim ... BSL --neg_num NEG_NUM --tau1 TAU1 --tau2 TAU2 ...
    ```
    where
    - `--neg_num` (int, optional, default=1000)
        The number of negative items for each user.
    - `--tau1` (float, optional, default=1.0)
        The temperature for the positive items.
    - `--tau2` (float, optional, default=1.0)
        The temperature for the negative items.
    
    ### GuidedRec
    ```
    itemrec ... optim ... GuidedRec --neg_num NEG_NUM
    ```
    where
    - `--neg_num` (int, optional, default=9)
        The number of negative items for each user.

    ### LambdaRank
    ```
    itemrec ... optim ... LambdaRank ...
    ```
    No additional arguments required.
    The `--neg_num` is set to constant 1 and should not be changed.

    ### LambdaLoss
    ```
    itemrec ... optim ... LambdaLoss ...
    ```
    No additional arguments required.
    The `--neg_num` is set to constant 1 and should not be changed.

    ### LambdaLossAtK
    ```
    itemrec ... optim ... LambdaLossAtK --k K ...
    ```
    where
    - `--k` (int, optional, default=20)
        The value of $K$ in LambdaLoss@K.

    ### LLPAUC
    ```
    itemrec ... optim ... LLPAUC --neg_num NEG_NUM --alpha ALPHA --beta BETA ...
    ```
    where
    - `--alpha` (float, optional, default=0.7)
        The alpha parameter for the LLPAUC optimizer.
    - `--beta` (float, optional, default=0.1)
        The beta parameter for the LLPAUC optimizer.

    ### PSL
    ```
    itemrec ... optim ... PSL --neg_num NEG_NUM --tau TAU --tau_star TAU_STAR
    --method METHOD --activation ACTIVATION ...
    ```
    where
    - `--neg_num` (int, optional, default=1000)
        The number of negative items for each user.
    - `--tau` (float, optional, default=2.0)
        The temperature parameter for the score function.
    - `--tau_star` (float, optional, default=1.0)
        The temperature parameter for the robustness.
    - `--method` (int, optional, default=1)
        The id of the PSL method, must be one of
        - 1: Softmax-like PSL
        - 2: BPR-like PSL
    - `--activation` (str, optional, default="tanh")
        The id of the activation function, must be one of
        - "tanh": $\log (\tanh(x) + 1)$
        - "relu": $\log (\text{ReLU}(x + 1))$
        - "atan": $\log (\arctan(x) + 1)$

    ### SLatK
    ```
    itemrec ... optim ... SLatK --neg_num NEG_NUM --tau TAU --tau_beta TAU_BETA 
    --k K --epoch_quantile EPOCH_QUANTILE ...
    ```
    where
    - `--neg_num` (int, optional, default=1000)
        The number of negative items for each user.
    - `--tau` (float, optional, default=1.0)
        The temperature for the softmax function.
    - `--tau_beta` (float, optional, default=1.0)
        The temperature for the softmax weights.
    - `--k` (int, optional, default=20)
        the Top-$K$ value.
    - `--epoch_quantile` (int, optional, default=20)
        The epoch interval for the quantile regression.

    ### Softmax
    ```
    itemrec ... optim ... Softmax --neg_num NEG_NUM --tau TAU ...
    ```
    where
    - `--neg_num` (int, optional, default=1000)
        The number of negative items for each user.
    - `--tau` (float, optional, default=1.0)
        The temperature for the softmax function.

    ## Returns
    - parser: argparse.ArgumentParser
        The sub/stage parser for ItemRec optimizer.
    """
    parser = argparse.ArgumentParser(
        description="The optim sub-command for ItemRec.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="The learning rate for the optimizer.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="The weight decay for the optimizer.",
    )
    optim_subparsers = parser.add_subparsers(
        title="optim sub-commands",
        description="The sub-commands for the optim sub-command.",
        dest="optim",
    )
    # optim: AdvInfoNCE
    advinfonce_parser = optim_subparsers.add_parser(
        "AdvInfoNCE",
        help="The AdvInfoNCE optimizer for ItemRec.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    advinfonce_parser.add_argument(
        "--neg_num",
        type=int,
        default=1000,
        help="The number of negative samples.",
    )
    advinfonce_parser.add_argument(
        "--tau",
        type=float,
        default=1.0,
        help="The temperature parameter.",
    )
    advinfonce_parser.add_argument(
        "--neg_weight",
        type=float,
        default=64,
        help="The negative weight.",
    )
    advinfonce_parser.add_argument(
        "--lr_adv",
        type=float,
        default=0.0001,
        help="The learning rate for adversarial learning.",
    )
    advinfonce_parser.add_argument(
        "--epoch_adv",
        type=int,
        default=1,
        help="The epoch interval for adversarial learning.",
    )
    # optim: BPR
    bpr_parser = optim_subparsers.add_parser(
        "BPR",
        help="The BPR optimizer for ItemRec.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    bpr_parser.add_argument(
        "--neg_num",
        type=int,
        default=1,
        help="The number of negative items for each user.",
    )
    # optim: GuidedRec
    guidedrec_parser = optim_subparsers.add_parser(
        "GuidedRec",
        help="The GuidedRec optimizer for ItemRec.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    guidedrec_parser.add_argument(
        "--neg_num",
        type=int,
        default=9,
        help="The number of negative items for each user.",
    )
    # optim: LambdaRank
    lambdarank_parser = optim_subparsers.add_parser(
        "LambdaRank",
        help="The LambdaRank optimizer for ItemRec.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    lambdarank_parser.add_argument(
        "--neg_num",
        type=int,
        default=1,
        help="The number of negative items for each user.",
    )
    # optim: LambdaLoss
    lambdaloss_parser = optim_subparsers.add_parser(
        "LambdaLoss",
        help="The LambdaLoss optimizer for ItemRec.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    lambdaloss_parser.add_argument(
        "--neg_num",
        type=int,
        default=1,
        help="The number of negative items for each user.",
    )
    # optim: LambdaLossAtK
    lambdalossatk_parser = optim_subparsers.add_parser(
        "LambdaLossAtK",
        help="The LambdaLossAtK optimizer for ItemRec.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    lambdalossatk_parser.add_argument(
        "--neg_num",
        type=int,
        default=1,
        help="The number of negative items for each user.",
    )
    lambdalossatk_parser.add_argument(
        "--k",
        type=int,
        default=20,
        help="The value of K in LambdaLoss@K.",
    )
    # optim: LLPAUC
    llpauc_parser = optim_subparsers.add_parser(
        "LLPAUC",
        help="The LLPAUC optimizer for ItemRec.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    llpauc_parser.add_argument(
        "--neg_num",
        type=int,
        default=1000,
        help="The number of negative items for each user.",
    )
    llpauc_parser.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="The alpha parameter of LLPAUC.",
    )
    llpauc_parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="The beta parameter of LLPAUC.",
    )
    # optim: BSL
    bsl_parser = optim_subparsers.add_parser(
        "BSL",
        help="The BSL optimizer for ItemRec.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    bsl_parser.add_argument(
        "--neg_num",
        type=int,
        default=1000,
        help="The number of negative items for each user.",
    )
    bsl_parser.add_argument(
        "--tau1",
        type=float,
        default=1.0,
        help="The temperature parameter for the positive items.",
    )
    bsl_parser.add_argument(
        "--tau2",
        type=float,
        default=1.0,
        help="The temperature parameter for the negative items.",
    )
    # optim: PSL
    psl_parser = optim_subparsers.add_parser(
        "PSL",
        help="The PSL optimizer for ItemRec.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    psl_parser.add_argument(
        "--neg_num",
        type=int,
        default=1000,
        help="The number of negative items for each user.",
    )
    psl_parser.add_argument(
        "--tau",
        type=float,
        default=2.0,
        help="The temperature parameter for the score function.",
    )
    psl_parser.add_argument(
        "--tau_star",
        type=float,
        default=1.0,
        help="The temperature parameter for the robustness.",
    )
    psl_parser.add_argument(
        "--method",
        type=int,
        default=1,
        help="The id of the PSL method.",
    )
    psl_parser.add_argument(
        "--activation",
        type=str,
        default="tanh",
        help="The id of the activation function.",
    )
    # optim: SLatK
    slatk_parser = optim_subparsers.add_parser(
        "SLatK",
        help="The SLatK optimizer for ItemRec.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    slatk_parser.add_argument(
        "--neg_num",
        type=int,
        default=1000,
        help="The number of negative items for each user.",
    )
    slatk_parser.add_argument(
        "--tau",
        type=float,
        default=1.0,
        help="The temperature for the softmax function.",
    )
    slatk_parser.add_argument(
        "--tau_beta",
        type=float,
        default=1.0,
        help="The temperature for the softmax weights.",
    )
    slatk_parser.add_argument(
        "--k",
        type=int,
        default=20,
        help="The Top-K value.",
    )
    slatk_parser.add_argument(
        "--epoch_quantile",
        type=int,
        default=20,
        help="The epoch interval for the quantile regression.",
    )
    # optim: Softmax
    softmax_parser = optim_subparsers.add_parser(
        "Softmax",
        help="The Softmax optimizer for ItemRec.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    softmax_parser.add_argument(
        "--neg_num",
        type=int,
        default=1000,
        help="The number of negative items for each user.",
    )
    softmax_parser.add_argument(
        "--tau",
        type=float,
        default=1.0,
        help="The temperature for the softmax function.",
    )
    return parser

def merge_args(args1: argparse.Namespace, args2: argparse.Namespace) -> argparse.Namespace:
    r"""
    ## Function
    Merge two parsed arguments into one.

    ## Arguments
    - args1: argparse.Namespace
        The first parsed arguments.
    - args2: argparse.Namespace
        The second parsed arguments.

    ## Returns
    - args: argparse.Namespace
        The merged parsed arguments.
    """
    args = argparse.Namespace()
    for k, v in args1.__dict__.items():
        setattr(args, k, v)
    for k, v in args2.__dict__.items():
        setattr(args, k, v)
    return args

# check arguments ---------------------------------------------------
def check_args(args: argparse.Namespace) -> None:
    r"""
    ## Function
    Check the validity of the parsed arguments.

    ## Arguments
    - args: argparse.Namespace
      - The parsed arguments.
    """
    # for BPR, LambdaRank, and LambdaLoss, the neg_num should be 1
    if args.optim == "BPR":
        assert args.neg_num == 1, f"For BPR, the neg_num should be 1, but got {args.neg_num}."
    if args.optim == "LambdaRank":
        assert args.neg_num == 1, f"For LambdaRank, the neg_num should be 1, but got {args.neg_num}."
    if args.optim == "LambdaLoss":
        assert args.neg_num == 1, f"For LambdaLoss, the neg_num should be 1, but got {args.neg_num}."
    if args.optim == "LambdaLossAtK":
        assert args.neg_num == 1, f"For LambdaLossAtK, the neg_num should be 1, but got {args.neg_num}."
    # for NCF, the layers should be a list of integers, and the last layer should be emb_size
    if args.model == "NCF":
        args.layers = eval(args.layers)
        assert isinstance(args.layers, list), f"For NCF, the layers should be a list of integers, but got {args.layers}."
        assert args.layers[-1] == args.emb_size, f"For NCF, the last layer should be emb_size, but got {args.layers[-1]}."
