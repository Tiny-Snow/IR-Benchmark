# IR Benchmark OOD Dataset Summary

## Introduction

This dataset summary provides a brief introduction to the Out-of-Distribution (OOD) datasets in our IR Benchmark. In general, the OOD datasets have different training and testing distributions, which can be used to evaluate the model's robustness. For convenience, all the OOD datasets are split into training and testing sets (no validation set). The training set remains the same as the original dataset, while the testing set is constructed by the specified OOD method.

We highly recommend you to read the survey paper [Bias and Debias in Recommender System: A Survey and Future Directions](https://jiawei-chen.github.io/paper/bias.pdf), after that you will have a better understanding of the OOD setting we list here: 

### OOD: Popularity Shift

The long-tail distribution is a common phenomenon in recommendation systems, where a few items are very popular while most items are rarely rated. In popularity shift, we use the long-tail training set to train the model, and test the model on the uniform testing set. 

In practice, the IID test set is constructed by randomly sampling the items in each user's positive interactions. However, to construct the OOD popularity shift test set, we first randomly choose the item, than randomly choose the user who has interacted with this item and select this user-item pair as the test interaction, until the test set size reaches 20% of the total interactions. The remaining 80% interactions are used as the training set. Some interactions may be discarded in test set to make sure all users and items in the test set are also occurred in the training set.

### OOD: Temporal Shift

In the real-world recommendation system, the user's preferences may change over time. In temporal shift, we use the historical training set to train the model, and test the model on the future testing set, which is just in line with the real-world recommendation scenario.

In practice, we construct the OOD temporal shift test set by its timestamp. We first sort the interactions by timestamp, then select the first 60% interactions as the training set and the last 20% interactions as the testing set. Some interactions may be discarded in test set to make sure all users and items in the test set are also occurred in the training set.

### OOD: Exposure Shift

Exposure bias happens as users are only exposed to a part of specific items so that unobserved interactions do not always represent negative preference. Exposure shift widely exists in the recommendation system, mainly caused by the strategy of the recommendation algorithm or the user activity.

In practice, the exposure shift is a nature of the OOD dataset and do not need to be re-constructed. We just use the original IID dataset as the exposure shift OOD dataset.


------

## Current Supported OOD Datasets

| Amazon2014 | Book Statistics | CD Statistics | Electronic Statistics | Movie Statistics |
| ---------- | ---------------- | ------------- | --------------------- | ---------------- |
| Users | 135,109 | 12,784 | 13,455 | 26,968 |
| Items | 115,172 | 13,874 | 8,360 | 18,563 |
| Interactions | 2,681,234 | 228,169 | 165,380 | 493,828 |
| Train Interactions | 2,425,429 | 216,457 | 140,712 | 457,774 |
| Test Interactions | 255,805 | 11,712 | 24,668 | 36,054 |
| Density | 0.00017 | 0.00129 | 0.00147 | 0.00099 |
| $N$-core | 10 | 10 | 10 | 10 |
| Rating | $\geq$ 3 | $\geq$ 3 | $\geq$ 3 | $\geq$ 3 |
| OOD Shift | Temporal | Temporal | Temporal | Temporal |
---

| Dataset | Users | Items | Interactions | Train Interactions | Test Interactions | Density | $N$-core | Rating | OOD Shift |
| ------- | ----- | ----- | ------------ | ------------------ | ----------------- | ------- | -------- | ---------------- | ----- |
| [Amazon-Book](#amazon) | 135,109 | 115,172 | 4,042,382 | 3,233,905 | 808,477 | 0.00026 | 10 | $\geq$ 3 | Popularity |
| [Amazon-Book](#amazon) | 135,109 | 115,172 | 2,681,234 | 2,425,429 | 255,805 | 0.00017 | 10 | $\geq$ 3 | Temporal |
| [Amazon-CD](#amazon) | 12,784 | 13,874 | 360,763 | 288,610 | 72,153 | 0.00203 | 10 | $\geq$ 3 | Popularity |
| [Amazon-CD](#amazon) | 12,784 | 13,874 | 228,169 | 216,457 | 11,712 | 0.00129 | 10 | $\geq$ 3 | Temporal |
| [Amazon-Electronic](#amazon) | 13,455 | 8,360 | 234,521 | 187,616 | 46,905 | 0.00208 | 10 | $\geq$ 3 | Popularity |
| [Amazon-Electronic](#amazon) | 13,455 | 8,360 | 165,380 | 140,712 | 24,668 | 0.00147 | 10 | $\geq$ 3 | Temporal |
| [Amazon-Movie](#amazon) | 26,968 | 18,563 | 762,957 | 610,365 | 152,592 | 0.00152 | 10 | $\geq$ 3 | Popularity |
| [Amazon-Movie](#amazon) | 26,968 | 18,563 | 493,828 | 457,774 | 36,054 | 0.00099 | 10 | $\geq$ 3 | Temporal |
| [Douban-Movie](#douban) | 51,803 | 25,993 | 8,355,102 | 6,684,101 | 1,671,001 | 0.00620 | 10 | $\geq$ 3 | Popularity |
| [Douban-Book](#douban) | 18,662 | 15,207 | 938,173 | 750,538 | 187,635 | 0.00331 | 10 | $\geq$ 3 | Popularity |
| [Douban-Music](#douban) | 13,343 | 15,957 | 903,178 | 722,542 | 180,636 | 0.00424 | 10 | $\geq$ 3 | Popularity |
| [Food](#food) | 5,875 | 9,852 | 170,253 | 139,822 | 30,431 | 0.00294 | 10 | $\geq$ 3 | Temporal |
| [Gowalla](#gowalla) | 29,858 | 40,988 | 1,027,464 | 821,971 | 205,493 | 0.00084 | 10 | $-$ | Popularity |
| [MovieLens-1M](#movielens) | 6,033 | 3,123 | 544,490 | 500,669 | 43,821 | 0.02890 | 10 | $\geq$ 3 | Temporal |
| [Yahoo Music](#yahoo-music) | 15,400 | 1,000 | 365,704 | 286,540 | 79,164 | 0.02375 | 10 | $-$ | Exposure |
| [Yelp2018](#yelp) | 55,616 | 34,945 | 1,506,777 | 1,205,421 | 301,356 | 0.00078 | 10 | $\geq$ 3 | Popularity |
| [Yelp2022](#yelp) | 69,090 | 41,814 | 1,897,447 | 1,517,957 | 379,490 | 0.00066 | 10 | $\geq$ 3 | Popularity |

------

### Amazon

#### Description

We processed [Amazon](https://github.com/Tiny-Snow/IR-Benchmark/blob/main/data/dataset_summary.md#amazon) dataset in the OOD Popularity Shift setting. For more details, you can refer to the [IID Amazon dataset](https://github.com/Tiny-Snow/IR-Benchmark/blob/main/data/dataset_summary.md#amazon).

Currently, we only support the following Amazon datasets in the OOD Popularity Shift and Temporal Shift setting:
- Amazon-Book
- Amazon-CD
- Amazon-Electronic
- Amazon-Movie

#### Final Dataset

| Amazon2014 | Book Statistics | CD Statistics | Electronic Statistics | Movie Statistics |
| ---------- | ---------------- | ------------- | --------------------- | ---------------- |
| Users | 135,109 | 12,784 | 13,455 | 26,968 |
| Items | 115,172 | 13,874 | 8,360 | 18,563 |
| Interactions | 4,042,382 | 360,763 | 234,521 | 762,957 |
| Train Interactions | 3,233,905 | 288,610 | 187,616 | 610,365 |
| Test Interactions | 808,477 | 72,153 | 46,905 | 152,592 |
| Density | 0.00026 | 0.00203 | 0.00208 | 0.00152 |
| $N$-core | 10 | 10 | 10 | 10 |
| Rating | $\geq$ 3 | $\geq$ 3 | $\geq$ 3 | $\geq$ 3 |
| OOD Shift | Popularity | Popularity | Popularity | Popularity |

| Amazon2014 | Book Statistics | CD Statistics | Electronic Statistics | Movie Statistics |
| ---------- | ---------------- | ------------- | --------------------- | ---------------- |
| Users | 135,109 | 12,784 | 13,455 | 26,968 |
| Items | 115,172 | 13,874 | 8,360 | 18,563 |
| Interactions | 2,681,234 | 228,169 | 165,380 | 493,828 |
| Train Interactions | 2,425,429 | 216,457 | 140,712 | 457,774 |
| Test Interactions | 255,805 | 11,712 | 24,668 | 36,054 |
| Density | 0.00017 | 0.00129 | 0.00147 | 0.00099 |
| $N$-core | 10 | 10 | 10 | 10 |
| Rating | $\geq$ 3 | $\geq$ 3 | $\geq$ 3 | $\geq$ 3 |
| OOD Shift | Temporal | Temporal | Temporal | Temporal |

------

### Douban

#### Description

We processed [Douban](https://github.com/Tiny-Snow/IR-Benchmark/blob/main/data/dataset_summary.md#douban) dataset in the OOD Popularity Shift setting. For more details, you can refer to the [IID Douban dataset](https://github.com/Tiny-Snow/IR-Benchmark/blob/main/data/dataset_summary.md#douban).

#### Final Dataset

| Douban | Movie Statistics | Book Statistics | Music Statistics |
| ------ | ---------------- | ---------------- | --------------- |
| Users | 51,803 | 18,662 | 13,343 |
| Items | 25,993 | 15,207 | 15,957 |
| Interactions | 8,355,102 | 938,173 | 903,178 |
| Train Interactions | 6,684,101 | 750,538 | 722,542 |
| Test Interactions | 1,671,001 | 187,635 | 180,636 |
| Density | 0.00620 | 0.00331 | 0.00424 |
| $N$-core | 10 | 10 | 10 |
| Rating | $\geq$ 3 | $\geq$ 3 | $\geq$ 3 |
| OOD Shift | Popularity | Popularity | Popularity |

------

### Food

#### Description

We processed [Food](https://github.com/Tiny-Snow/IR-Benchmark/blob/main/data/dataset_summary.md#food) dataset in the OOD Temporal Shift setting. For more details, you can refer to the [IID Food dataset](https://github.com/Tiny-Snow/IR-Benchmark/blob/main/data/dataset_summary.md#food).

#### Final Dataset

| Food | Statistics |
| ---- | ---------- |
| Users | 5,875 |
| Items | 9,852 |
| Interactions | 170,253 |
| Train Interactions | 139,822 |
| Test Interactions | 30,431 |
| Density | 0.00294 |
| $N$-core | 10 |
| Rating | $\geq$ 3 |
| OOD Shift | Temporal |

------

### Gowalla

#### Description

We processed [Gowalla](https://github.com/Tiny-Snow/IR-Benchmark/blob/main/data/dataset_summary.md#gowalla) dataset in the OOD Popularity Shift setting. For more details, you can refer to the [IID Gowalla dataset](https://github.com/Tiny-Snow/IR-Benchmark/blob/main/data/dataset_summary.md#gowalla).

#### Final Dataset

| Gowalla | Statistics |
| ----- | ---------- |
| Users | 29,858 |
| Items | 40,988 |
| Interactions | 1,027,464 |
| Train Interactions | 821,971 |
| Test Interactions | 205,493 |
| Density | 0.00084 |
| $N$-core | 10 |
| OOD Shift | Popularity |

------

### MovieLens

#### Description

We processed [MovieLens](https://github.com/Tiny-Snow/IR-Benchmark/blob/main/data/dataset_summary.md#movielens) dataset in the OOD Temporal Shift setting. For more details, you can refer to the [IID MovieLens dataset](https://github.com/Tiny-Snow/IR-Benchmark/blob/main/data/dataset_summary.md#movielens).

Currently, we only support the MovieLens-1M dataset in the OOD Temporal Shift setting. Our aim is to process all the MovieLens datasets in the OOD Temporal Shift setting in the future.

#### Final Dataset

| MovieLens | Statistics |
| --------- | ---------- |
| Users | 6,033 |
| Items | 3,123 |
| Interactions | 544,490 |
| Train Interactions | 500,669 |
| Test Interactions | 43,821 |
| Density | 0.02890 |
| $N$-core | 10 |
| Rating | $\geq$ 3 |
| OOD Shift | Temporal |

------

### Yahoo Music

#### Description

We processed [Yahoo Music](https://github.com/Tiny-Snow/IR-Benchmark/blob/main/data/dataset_summary.md#yahoo-music) dataset in the OOD Exposure Shift setting. For more details, you can refer to the [IID Yahoo Music dataset](https://github.com/Tiny-Snow/IR-Benchmark/blob/main/data/dataset_summary.md#yahoo-music).

#### Final Dataset

| Yahoo Music | Statistics |
| ----------- | ---------- |
| Users | 15,400 |
| Items | 1,000 |
| Interactions | 365,704 |
| Train Interactions | 286,540 |
| Test Interactions | 79,164 |
| Density | 0.02375 |
| $N$-core | 10 |
| Rating | $-$ |
| OOD Shift | Exposure |

------

### Yelp

#### Description

We processed [Yelp](https://github.com/Tiny-Snow/IR-Benchmark/blob/main/data/dataset_summary.md#yelp) dataset in the OOD Popularity Shift setting. For more details, you can refer to the [IID Yelp dataset](https://github.com/Tiny-Snow/IR-Benchmark/blob/main/data/dataset_summary.md#yelp).

#### Final Dataset

| Yelp | Yelp2018 Statistics | Yelp2022 Statistics |
| ----- | ---------- | ---------- |
| Users | 55,616 | 69,090 |
| Items | 34,945 | 41,814 |
| Interactions | 1,506,777 | 1,897,447 |
| Train Interactions | 1,205,421 | 1,517,957 |
| Test Interactions | 301,356 | 379,490 |
| Density | 0.00078 | 0.00066 |
| $N$-core | 10 | 10 |
| Rating | $\geq$ 3 | $\geq$ 3 |
| OOD Shift | Popularity | Popularity |

------

