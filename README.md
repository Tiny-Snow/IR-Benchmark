# Item Recommendation Benchmark


## Introduction

**IR-Benchmark** is a unified, extensible, and reproducible benchmark for **collaborative filtering (CF) research**, including:
- **Benchmark datasets**: various IID and OOD (with popularity bias) recommendation datasets, e.g., Amazon, Douban, Gowalla, MovieLens, Yelp, etc.
- **Advanced recommendation models**: conventional and advanced recommendation models, e.g., MF, LightGCN, LightGCN++, XSimGCL, etc.
- **SOTA recommendation losses**: various state-of-the-art recommendation losses, e.g., BPR, Lambdaloss, SL, PSL, SL@$K$, etc.
- **Unified interface**: a unified interface for data processing, training, evaluation, and hyperparameter tuning based on [NNI framework](https://github.com/microsoft/nni).
- **Easy-to-extend**: decoupling structure for easy extension of new datasets, models, and losses.

## :tada: News

- **[May 28, 2025]** **IR-Benchmark V1.0** has been released!
- **[May 15, 2025]** Our paper [Breaking the Top-$K$ Barrier: Advancing Top-$K$ Ranking Metrics Optimization in Recommender Systems](https://openreview.net/forum?id=5hKTOc3ch1), which proposes the SL@$K$ loss, has been accepted to **SIGKDD 2025** with **Novelty 3.6/4.0** and **Technical Quality 4.0/4.0**!
- **[Sep 26, 2024]** Our paper [PSL: Rethinking and Improving Softmax Loss from Pairwise Perspective for Recommendation](https://arxiv.org/abs/2411.00163), which proposes the PSL loss, has been accepted to **NeurIPS 2024**!

## Benchmark Datasets

We provide a variety of recommendation datasets in [Tiny-Snow/IR-Benchmark-Dataset](https://github.com/Tiny-Snow/IR-Benchmark-Dataset), including both IID and OOD datasets. Please refer to the [IID dataset summary](https://github.com/Tiny-Snow/IR-Benchmark-Dataset/blob/main/data_iid/dataset_summary.md) and [OOD dataset summary](https://github.com/Tiny-Snow/IR-Benchmark-Dataset/blob/main/data_ood/dataset_summary_ood.md) for more details.

Additionally, we provide a unified interface for performance evaluation, supporting a wide range of evaluation metrics, including NDCG@$K$, Recall@$K$, Precision@$K$, MRR@$K$, HitRatio@$K$, F1@$K$ and AUC.

## Benchmark Models

We provide multiple recommendation models to facilitate the research:

| Model | Description | Paper |
| :--- | :--- | :--- |
| [MF](src/itemrec/model/model_MF.py) | Basic Matrix Factorization model | [Koren et al., Computer '09](https://ieeexplore.ieee.org/abstract/document/5197422) |
| [NCF](src/itemrec/model/model_NCF.py) | Neural Collaborative Filtering model | [He et al., WWW '17](https://dl.acm.org/doi/10.1145/3038912.3052569) |
| [LightGCN](src/itemrec/model/model_LightGCN.py) | Simplified GCN model for recommendation | [He et al., SIGIR '20](https://doi.org/10.1145/3397271.3401063) |
| [SimpleX](src/itemrec/model/model_SimpleX.py) | A simple MF with historical interactions awareness | [Mao et al., CIKM '21](https://dl.acm.org/doi/10.1145/3459637.3482297) |
| [SimGCL](src/itemrec/model/model_SimGCL.py) | Simple Graph Contrastive Learning model based on LightGCN | [Yu et al., TKDE '24](https://ieeexplore.ieee.org/abstract/document/10158930) |
| [XSimGCL](src/itemrec/model/model_XSimGCL.py) | eXtremely Simple Graph Contrastive Learning model based on LightGCN | [Yu et al., TKDE '24](https://ieeexplore.ieee.org/abstract/document/10158930) |
| [LightGCN++](src/itemrec/model/model_LightGCNpp.py) | Enhanced LightGCN with additional normalization | [Lee et al., RecSys '24](https://dl.acm.org/doi/10.1145/3640457.3688176) |

## Benchmark Losses

We provide various recommendation losses to fill the gap where the existing repositories only support conventional losses like BPR and SL:

| Loss | Description | Paper |
| :--- | :--- | :--- |
| [BPR](src/itemrec/optimizer/optim_BPR.py) | Bayesian Personalized Ranking loss, the conventional pairwise loss | [Rendle et al., UAI '09](https://arxiv.org/abs/1205.2618) |
| [LambdaRank](src/itemrec/optimizer/optim_LambdaRank.py) | A conventional pairwise loss for ranking | [Burges, Learning '10](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf) |
| [LambdaLoss](src/itemrec/optimizer/optim_LambdaLoss.py) | A variant of LambdaRank for NDCG optimization | [Wang et al., CIKM '18](https://dl.acm.org/doi/abs/10.1145/3269206.3271784) |
| [GuidedRec](src/itemrec/optimizer/optim_GuidedRec.py) | A model surrogate method for NDCG | [Rashed et al., SIGIR'21](https://dl.acm.org/doi/abs/10.1145/3404835.3462864) |
| [LambdaLoss@$K$](src/itemrec/optimizer/optim_LambdaLossAtK.py) | A variant of LambdaLoss for NDCG@$K$ optimization | [Jagerman et al., SIGIR '22](https://dl.acm.org/doi/abs/10.1145/3477495.3531849) |
| [SogCLR](src/itemrec/optimizer/optim_SogCLR.py) | A variant of SimCLR/SL for small-batch negative sampling | [Yuan et al., ICML '22](https://proceedings.mlr.press/v162/yuan22b.html) |
| [AdvInfoNCE](src/itemrec/optimizer/optim_AdvInfoNCE.py) | Adversarial InfoNCE loss, a DRO variant of InfoNCE/SL | [Zhang et al., NeurIPS '23](https://proceedings.neurips.cc/paper_files/paper/2023/hash/13f1750b825659394a6499399e7637fc-Abstract-Conference.html) |
| [SL](src/itemrec/optimizer/optim_Softmax.py) | Softmax loss, the SOTA cross-entropy loss for recommendation | [Wu et al., TOIS '24](https://dl.acm.org/doi/10.1145/3637061) |
| [LLPAUC](src/itemrec/optimizer/optim_LLPAUC.py) | Lower-Left Partial AUC loss, a surrogate loss for Recall@$K$ and Precision@$K$ | [Shi et al., WWW '24](https://dl.acm.org/doi/10.1145/3589334.3645371) |
| [BSL](src/itemrec/optimizer/optim_BSL.py) | Bilateral Softmax Loss, a DRO variant of SL | [Wu et al., ICDE '24](https://ieeexplore.ieee.org/abstract/document/10598015) |
| [PSL](src/itemrec/optimizer/optim_PSL.py) | Pairwise Softmax Loss, a pairwise extension of SL, which only changes the activation function | [Yang et al., NeurIPS '24](https://arxiv.org/abs/2411.00163) <br> **(Ours)** |
| [SL@$K$](src/itemrec/optimizer/optim_SLatK.py) | The SOTA surrogate loss for NDCG@$K$, which is essentially a weighted SL| [Yang et al., KDD '25](https://openreview.net/forum?id=5hKTOc3ch1) <br> **(Ours)** |
| [SogSL@$K$](src/itemrec/optimizer/optim_SogSLatK.py) | A SogCLR-enhanced variant of SL@$K$ for small-batch negative sampling | [Yuan et al., ICML '22](https://proceedings.mlr.press/v162/yuan22b.html) <br> [Yang et al., KDD '25](https://openreview.net/forum?id=5hKTOc3ch1) <br> **(Ours)** |

## :rocket: Quick Start

### Environment Setup

The environment is provided in the [environment.yml](environment.yml) file. You can create a conda environment `itemrec` with the following command:

```bash
conda env create -f environment.yml
```

Some issues may occur when installing `torch` as well as other torch-related packages. If you encounter any issues, please install them manually. After the environment is created, you can activate it with:

```bash
conda activate itemrec
```

### CLI Usage

A CLI interface is provided for IR-Benchmark, which allows you to run the benchmark with a single command in the following structure:

```bash
python -u -m itemrec [-h] [-v] --log LOG --save_dir SAVE_DIR --seed SEED \
  model [--model_args ...] dataset [--dataset_args ...] optim [--optim_args ...]
```

where `model`, `dataset`, and `optim` are the subcommands to specify the model, dataset, and optimization algorithm, respectively. We adopt a decoupled structure for each component, which allows you combine any model, dataset, and loss function together flexibly. An example command is as follows:

```bash
python -u -m itemrec --log=/path/to/ir.log --save_dir=/path/to/save/dir --seed=2024 \
  model --emb_size=64 --norm --num_epochs=200 MF \
  dataset --data_path=/path/to/data/amazon2014-health/proc --batch_size=1024 --num_workers=16 --sampler=uniform --epoch_sampler=0 --fold=5 \
  optim --lr=0.01 --weight_decay=0.0 SLatK --neg_num=1000 --tau=0.2 --tau_beta=2.25 --k=20 --epoch_quantile=20
```

The above command configures the following settings:
- The log file and model will be saved to `save_dir`, which is `/path/to/save/dir`.
- The MF model is used as backbone, where the embedding size `emb_size` is set to 64, and the cosine similarity is used for embedding normalization (`--norm`).
- The dataset path is the Amazon2014-Health in `data_path`, i.e., `/path/to/data/amazon2014-health/proc`.
- The number of epochs is set to 200, with a batch size of 1024 and 16 workers for data loading. The negative sampler is set to `uniform` sampling (i.e., randomly sampling items except for the positive item). The `epoch_sampler` is used for updating the sampler every `epoch_sampler` epochs, which is useful for other samplers (e.g., hard-negative sampling). We use 5-fold cross-validation (`fold=5`).
- The loss function is set to SL@$K$ with $K=20$, a negative sampling number `neg_num` of 1000, a temperature `tau` of 0.2, a `tau_beta` value of 2.25, and a quantile update period `epoch_quantile` of 20 epochs. The learning rate `lr` is set to 0.01, and the weight decay `weight_decay` is set to 0.0.

For detailed CLI usage, please refer to [src/itemrec/args.py](src/itemrec/args.py).

### NNI Hyperparameter Tuning

For more quick start and result reproduction, we recommend using the [NNI framework](https://github.com/microsoft/nni) provided in [src/run_nni.py](src/run_nni.py). Specifically, we provide a pre-defined CLI command as above in [src/run_nni.py](src/run_nni.py), and the hyperparameters are specified in [src/itemrec/hyper.py](src/itemrec/hyper.py).

For quick start, if we want to test SL@$K$ on 2-layer LightGCN and Gowalla dataset, we first set the following paths in [src/run_nni.py](src/run_nni.py) to your own paths:

```python
# TODO: `/path/to/your/` must be replaced with the actual paths
save_dir = f"/path/to/your/logs/{args.dataset}/{args.model}/{args.optim}"
if not args.ood:
    dataset_path = f"/path/to/your/data/{args.dataset}/proc"
else:
    dataset_path = f"/path/to/your/data_ood/{args.dataset}/proc"
...
experiment.config.trial_code_directory = '/path/to/your/src'
```

as well as the NNI configurations, e.g., the `max_trial_number_per_gpu` and `gpu_indices`, which specify the number of trials per GPU and the GPU indices to use:

```python
experiment.config.training_service.max_trial_number_per_gpu = 2
experiment.config.training_service.gpu_indices = [0, 1, 2, 3]
```

Then, specify the hyperparameter search space in [src/itemrec/hyper.py](src/itemrec/hyper.py). An example is as follows:

```python
search_space_dict = {
    'SLatK': {
        'lr': {'_type': 'choice', '_value': [0.1, 0.01, 0.001]},
        'weight_decay': {'_type': 'choice', '_value': [0, 1e-4, 1e-5, 1e-6]},
        'tau': {'_type': 'choice', '_value': [0.01, 0.025, 0.05, 0.1, 0.2, 0.5]},
        'tau_beta': {'_type': 'quniform', '_value': [0.5, 3, 0.25]},
        'k': {'_type': 'choice', '_value': [5, 20, 50]},
        'epoch_quantile': {'_type': 'choice', '_value': [5, 20]},
    },
    ...
}
```

Finally, run the following command to start the NNI experiment at `10032` port, with no cross-validation (`fold=1`):

```bash
python run_nni.py --model=LightGCN --num_layers=2 --dataset=gowalla --optim=SLatK --norm --fold=5 --port=10032
```

## :balance_scale: License

This software is provided under the [GPL-3.0 license](https://choosealicense.com/licenses/gpl-3.0/) © 2024 [Tiny Snow](https://tiny-snow.github.io/). All rights reserved.

## :thought_balloon: Feedback

This repository is initially built by [Tiny Snow](https://tiny-snow.github.io/) for research purpose, which is easily extensible. If you find any bugs or want to contribute to this repository, please feel free to open an issue or pull request.

## Citation

If you find this repository useful, please consider citing the following paper:

```bibtex
@inproceedings{yang2024psl,
  author = {Yang, Weiqin and Chen, Jiawei and Xin, Xin and Zhou, Sheng and Hu, Binbin and Feng, Yan and Chen, Chun and Wang, Can},
  booktitle = {Advances in Neural Information Processing Systems},
  editor = {A. Globerson and L. Mackey and D. Belgrave and A. Fan and U. Paquet and J. Tomczak and C. Zhang},
  pages = {120974--121006},
  publisher = {Curran Associates, Inc.},
  title = {PSL: Rethinking and Improving Softmax Loss from Pairwise Perspective for Recommendation},
  url = {https://proceedings.neurips.cc/paper_files/paper/2024/file/db1d5c63576587fc1d40d33a75190c71-Paper-Conference.pdf},
  volume = {37},
  year = {2024}
}
@inproceedings{yang2025breaking,
  title={Breaking the Top-\$K\$ Barrier: Advancing Top-\$K\$ Ranking Metrics Optimization in Recommender Systems},
  author={Yang, Weiqin and Chen, Jiawei and Zhang, Shengjia and Wu, Peng and Sun, Yuegang and Feng, Yan and Chen, Chun and Wang, Can},
  booktitle={31st SIGKDD Conference on Knowledge Discovery and Data Mining - Research Track (February 2025 Deadline)},
  year={2025},
}
```
