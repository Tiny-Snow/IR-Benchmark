# -------------------------------------------------------------------
# ItemRec / Item Recommendation Benchmark
# Copyright (C) 2024 Tiny Snow / Weiqin Yang @ Zhejiang University
# -------------------------------------------------------------------
# Module: NNI Main Entry
# Description:
#  This module is not a part of the ItemRec project, but it is used
#  to run the ItemRec project with NNI (Neural Network Intelligence).
#  If you want to run the ItemRec with NNI, you can directly run this
#  module.
#  NOTE: You can modify this code to fit your own project.
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
from itemrec.hyper import get_search_space
import argparse
import nni
from nni.experiment import Experiment

# public functions --------------------------------------------------
__all__ = [

]

# arguments ---------------------------------------------------------
def parse_args() -> argparse.Namespace:
    r"""
    Parse the arguments for the main function.
    """
    parser = argparse.ArgumentParser(description='ItemRec with NNI')
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='The name of the model'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='The name of the dataset'
    )
    parser.add_argument(
        '--optim',
        type=str,
        required=True,
        help='The name of the optimizer'
    )
    parser.add_argument(
        "--norm",
        action="store_true",
        help="Whether to normalize the embeddings. If true, the cosine similarity is used in the evaluation.",
    )
    parser.add_argument(
        "--ood",
        action="store_true",
        help="Whether to use the out-of-distribution test set."
    )
    parser.add_argument(
        "--neg_num", 
        type=int,
        default=1000,
        help="The number of negative samples"
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=200,
        help='The number of epochs'
    )
    parser.add_argument(
        '--num_layers',
        type=int,
        default=3,
        help='The number of layers of the LightGCN model'
    )
    parser.add_argument(
        '--contrast_weight', 
        type=float,
        default=0.2,
        help='The weight of the contrastive loss in XSimGCL'
    )
    parser.add_argument(
        '--method',
        type=int,
        default=1,
        help='The method of the PSL optimizer'
    )
    parser.add_argument(
        '--activation',
        type=str,
        default='tanh',
        help='The activation function of the PSL optimizer'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=20,
        help='The value of k in SLatK'
    )
    parser.add_argument(
        '--no_valid',
        action='store_true',
        help='Whether to use the validation set'
    )
    parser.add_argument(
        '--sampler',
        type=str,
        default='uniform',
        help='The sampler for negative sampling'
    )
    parser.add_argument(
        '--epoch_sampler',
        type=int,
        default=0,
        help='The number of epochs to update the sampler'
    )
    parser.add_argument(
        '--fold',
        type=int,
        default=1,
        help='The number of folds for cross-validation'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=16,
        help='The number of workers for the data loader'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8080,
        help='The port of the NNI experiment'
    )
    return parser.parse_args()

# main function -----------------------------------------------------
def main():
    args = parse_args()

    # TODO: `/path/to/your/` must be replaced with the actual paths
    save_dir = f"/path/to/your/logs/{args.dataset}/{args.model}/{args.optim}"
    if not args.ood:
        dataset_path = f"/path/to/your/data/{args.dataset}/proc"
    else:
        dataset_path = f"/path/to/your/data_ood/{args.dataset}/proc"
    norm_cmd = "--norm " if args.norm else ""
    if args.model == 'LightGCN':
        model_cmd = f"LightGCN --num_layers={args.num_layers} "
    elif args.model == 'LightGCNPP':
        model_cmd = f"LightGCNPP --num_layers={args.num_layers} "
    elif args.model == 'MF':
        model_cmd = f"MF "
    elif args.model == 'NCF':
        model_cmd = f"NCF "
    elif args.model == 'SimGCL':
        model_cmd = f"SimGCL --contrast_weight={args.contrast_weight} "
    elif args.model == 'SimpleX':
        model_cmd = f"SimpleX --history_len=50 --history_weight=0.5 "
    elif args.model == 'XSimGCL':
        model_cmd = f"XSimGCL --contrast_weight={args.contrast_weight} "
    else:
        raise ValueError(f"Invalid model: {args.model}")
    if args.optim == 'AdvInfoNCE':
        optim_cmd = f"AdvInfoNCE --neg_num={args.neg_num} --tau=0.2 --neg_weight=64 --lr_adv=5e-5 --epoch_adv=5 "
    elif args.optim == 'BPR':
        optim_cmd = f"BPR "
    elif args.optim == 'BSL':
        optim_cmd = f"BSL --neg_num={args.neg_num} --tau1=0.2 --tau2=0.2 "
    elif args.optim == 'GuidedRec':
        optim_cmd = f"GuidedRec --neg_num=9 "
    elif args.optim == 'LambdaRank':
        optim_cmd = f"LambdaRank "
    elif args.optim == 'LambdaLoss':
        optim_cmd = f"LambdaLoss "
    elif args.optim == 'LambdaLossAtK':
        optim_cmd = f"LambdaLossAtK --k={args.k} "
    elif args.optim == 'LLPAUC':
        optim_cmd = f"LLPAUC --neg_num={args.neg_num} --alpha=0.7 --beta=0.1 "
    elif args.optim == 'PSL':
        assert args.method in [1, 2]
        assert args.activation in ['tanh', 'relu', 'atan']
        optim_cmd = f"PSL --neg_num={args.neg_num} --tau=2.0 --tau_star=0.1 --method={args.method} --activation={args.activation} "
    elif args.optim == 'SLatK':
        optim_cmd = f"SLatK --neg_num={args.neg_num} --tau=0.2 --tau_beta=1.0 --k={args.k} --epoch_quantile=5 "
    elif args.optim == 'Softmax':
        optim_cmd = f"Softmax --neg_num={args.neg_num} --tau=0.2 "
    elif args.optim == 'SogCLR':
        optim_cmd = f"SogCLR --neg_num={args.neg_num} --tau=0.2 --gamma_g=0.9 "
    elif args.optim == 'SogSLatK':
        optim_cmd = f"SogSLatK --neg_num={args.neg_num} --tau=0.2 --tau_beta=1.0 --k={args.k} --epoch_quantile=5 --gamma_g=0.9 "
    else:
        raise ValueError(f"Invalid optimizer: {args.optim}")

    # ItemRec command
    cmd = f"python -u -m itemrec " \
        f"--log={save_dir}/ir.log " \
        f"--save_dir={save_dir} " \
        f"--seed=2024 " \
        f"model --emb_size=64 " + norm_cmd + f"--num_epochs={args.num_epochs} " + model_cmd + \
        f"dataset --data_path={dataset_path} --batch_size=1024 --num_workers={args.num_workers} " \
            + ("--no_valid " if args.ood or args.no_valid else "") \
            + f"--sampler={args.sampler} --epoch_sampler={args.epoch_sampler} --fold={args.fold} " + \
        f"optim --lr=1e-1 --weight_decay=0.0 " + optim_cmd
    
    # NNI experiment
    experiment = Experiment('local')
    experiment.config.experiment_name = 'ItemRec'

    experiment.config.trial_command = cmd
    experiment.config.trial_code_directory = '/path/to/your/src'
    experiment.config.trial_concurrency = 1000
    experiment.config.trial_gpu_number = 1

    experiment.config.training_service.platform = 'local'
    experiment.config.training_service.use_active_gpu = True
    experiment.config.training_service.max_trial_number_per_gpu = 2
    experiment.config.training_service.gpu_indices = [0, 1, 2, 3]

    experiment.config.search_space = get_search_space(args.optim)

    experiment.config.tuner.name = 'TPE'
    experiment.config.tuner.class_args = {
        'optimize_mode': 'maximize'
    }

    # NOTE: use the median stop if needed
    # experiment.config.assessor.name = 'Medianstop'
    # experiment.config.assessor.class_args = {
    #     'optimize_mode': 'maximize',
    #     'start_step' : args.num_epochs // 5,
    # }

    experiment.run(args.port)


if __name__ == '__main__':
    main()

