# IR Benchmark Dataset Summary

## Introduction

Our IR Benchmark aims to provide a unified data interface for Collaborative Filtering (CF). The dataset processing pipeline is integrated into our framework, users are only required to provide several hyperparameters. However, you should at least know the dataset format and what the data means here.

### Collaborative Filtering: A Brief Introduction

If you are the first time to hear about Collaborative Filtering (CF), you can refer to the [Wikipedia](https://en.wikipedia.org/wiki/Collaborative_filtering) for a brief introduction. We also highly recommend the survey paper [Bias Issues and Solutions in Recommender System](https://jiawei-chen.github.io/paper/biastutorial.pdf) for both beginners and experts. Here we will provide a brief introduction to CF mainly based on [Neural Collaborative Filtering](https://hexiangnan.github.io/papers/www17-ncf.pdf).

In CF, we name the users and items by their IDs, namely $u_1, u_2, \cdots, u_M$ and $i_1, i_2, \cdots, i_N$. The dataset consists of a set of interactions (user-item pair) between users and items, which can be represented as a matrix $R \in \mathbb{R}^{M \times N}$, $R_{ui} = 1$ if user $u$ has interacted with item $i$, in other words the user-item pair $(u, i)$ is in the dataset, otherwise $R_{ui} = 0$.

In CF, we only have positive data (i.e. the observed interactions), and our goal is to rank all the items $i_1, i_2, \cdots, i_N$ for each user, and select the Top-$K$ items as the recommendation list. The model (e.g. Neural Network) will be trained to predict a score $s_{ui}$ for each user-item pair $(u, i)$, and the ranking is based on these scores. 

The basic idea and assumption of CF is: similar users will have similar preferences, and similar items will be liked by similar users. Thus, the score $s_{ui}$ is typically defined as the similarity between user $u$ and item $i$, one common way is to use the inner product of user and item embeddings, i.e. $s_{ui} = \mathbf{u}_u^T \mathbf{i}_i$. The most famous CF models [Matrix Factorization (MF)](https://datajobs.com/data-science-repo/Recommender-Systems-%5BNetflix%5D.pdf), [Neural Graph Collaborative Filtering (NGCF)](https://dl.acm.org/doi/abs/10.1145/3331184.3331267), [LightGCN](https://dl.acm.org/doi/abs/10.1145/3397271.3401063), etc. are all based on this idea.

Now you may have a basic understanding of CF. 

### Basic Terminologies

- Implicit Feedback: In CF, the explicit feedback is the rating given by the user to the item, e.g. the number of stars for a movie. However, in many cases, we only have implicit feedback, i.e. the observed interactions, e.g. the user has clicked, purchased, or viewed the item. Our IR Benchmark mainly focuses on the implicit feedback. 
For explicit feedback datasets, we consider the interactions with a rating greater than a threshold (e.g. 3 in 1-5 rating) as positive/observed interactions. Another common practice is to mark all the observed interactions as positive, whether the rating is high or low. Currently, we only consider the threshold method.
- $N$-core: We only retain the users and items that have at least $N$ interactions. This is a common practice in CF, as it can filter out the users and items with very few interactions, which are not informative for the model training. Typically, $N$ is set to 10 here.
It should be noted that the $N$-core filtering may loop multiple times, each time we count the number of interactions for each user and item, i.e. `user_cnt[u]` and `item_cnt[i]`, and only retain the interactions with `user_cnt[u] >= N` and `item_cnt[i] >= N`. However, after one filtering, the `user_cnt` and `item_cnt` will be changed, and we need to re-calculate the `user_cnt` and `item_cnt` for the remaining interactions. We will loop until no interactions are removed.
- Top-$K$: We use the Top-$K$ items as the recommendation list for each user. We calculate the recommendation metrics (e.g. Recall@$K$, NDCG@$K$) based on the Top-$K$ items. Typically, $K$ is set to 20, 50, 100, etc.
- Train-Test Split: For each user, we randomly split the positive items (i.e. has interactions) into training and testing sets by the specified ratio, typically 8:2. The purpose of this split is to make sure the training and testing sets have the same users, since the recommendation is user-based. 
What's more, after train-test split, we will further filter the items in the testing set and only retain the items that also appear in the training set (however, it's nearly impossible that one item only appears in the testing set).
- Density and Sparsity: The density of the dataset is defined as the ratio of the number of interactions to the number of all possible interactions, i.e. $density = \frac{|\text{interactions}|}{M \times N}$. The sparsity is defined as $1 - density$. These two metrics are used to measure the informativeness of the dataset (or the user-item matrix). A sparse dataset means that most of the user-item pairs are not observed, which makes the recommendation task more challenging.


### Dataset Format and Cleaning

In our IR Benchmark, one dataset will have four files: 
``` # 用文件树表示
dataset_name
├── user_list.tsv       # user_orgid, user_procid
├── item_list.tsv       # item_orgid, item_procid
├── train.tsv           # user_procid, item_procid
├── test.tsv            # user_procid, item_procid
└── summary.txt         # dataset summary
```

We represent the user and item by their IDs (number). Since the original ID may be string, and it's a common practice to apply $N$-core filtering, we need to re-map the original IDs `orgid` to the processed IDs `procid`. The processed IDs will name from 0 and compress the original sparse IDs. The re-map dictionary `{orgid: procid}` will be saved in `user_list.tsv` and `item_list.tsv`.

After the re-mapping, we will randomly split the interactions `(user_procid, item_procid)` into training and testing sets. The training set will be saved in `train.tsv`, and the testing set will be saved in `test.tsv`. By default, the ratio of training and testing is 8:2. `train.tsv` and `test.tsv` are the only two files that you need to use for research purposes.

Finally, we will save a `summary.txt` file to record the dataset statistics and important hyperparameters. Here is an example: 
```
Dataset: Gowalla
Number of users: 29858
Number of items: 40988
Number of interactions: 1027464
Number of train interactions: 810128
Number of test interactions: 217336
Density: 0.00084
Number of unique users in train: 29858
Number of unique items in train: 40988
Number of unique users in test: 29858
Number of unique items in test: 39520
N-core: 10
```

------

## Current Supported Datasets

| Dataset | Users | Items | Interactions | Train Interactions | Test Interactions | Density | $N$-core | Rating |
| ------- | ----- | ----- | ------------ | ------------------ | ----------------- | ------- | -------- | ---------------- |
| [Amazon2014](#amazon) | 718,190 | 460,233 | 16,971,561 | 13,294,890 | 3,676,671 | 0.00005 | 10 | $\geq$ 3 |
| [Amazon2014-Beauty](#amazon) | 1,237 | 719 | 25,736 | 20,101 | 5,635 | 0.02894 | 10 | $\geq$ 3 |
| [Amazon2014-Book](#amazon) | 135,109 | 115,172 | 4,042,382 | 3,180,699 | 861,683 | 0.00026 | 10 | $\geq$ 3 |
| [Amazon2014-CD](#amazon) | 12,784 | 13,874 | 360,763 | 283,608 | 77,155 | 0.00203 | 10 | $\geq$ 3 |
| [Amazon2014-Electronic](#amazon) | 13,455 | 8,360 | 234,521 | 182,398 | 52,123 | 0.00208 | 10 | $\geq$ 3 |
| [Amazon2014-Health](#amazon) | 1,974 | 1,200 | 48,189 | 37,784 | 10,405 | 0.02034 | 10 | $\geq$ 3 |
| [Amazon2014-Movie](#amazon) | 26,968 | 18,563 | 762,957 | 599,903 | 163,054 | 0.00152 | 10 | $\geq$ 3 |
| [Amazon2018](#amazon) | 1,591,189 | 3,481,925 | 88,111,403 | 69,856,556 | 18,254,847 | 0.00002 | 10 | $\geq$ 3 |
| [Amazon2018-Book](#amazon) | 274,926 | 537,851 | 14,847,692 | 11,768,846 | 3,078,846 | 0.00010 | 10 | $\geq$ 3 |
| [Douban-Movie](#douban) | 51,803 | 25,993 | 8,355,127 | 6,664,958 | 1,690,169 | 0.00620 | 10 | $\geq$ 3 |
| [Douban-Book](#douban) | 18,662 | 15,207 | 938,173 | 743,085 | 195,088 | 0.00331 | 10 | $\geq$ 3 |
| [Douban-Music](#douban) | 13,343 | 15,957 | 903,178 | 717,188 | 185,990 | 0.00424 | 10 | $\geq$ 3 |
| [Food](#food) | 5,875 | 9,852 | 233,038 | 184,106 | 48,932 | 0.00403 | 10 | $\geq$ 3 |
| [Gowalla](#gowalla) | 29,858 | 40,988 | 1,027,464 | 810,128 | 217,336 | 0.00084 | 10 | $-$ |
| [Movielens-100k](#movielens) | 939 | 1,016 | 80,393 | 63,944 | 16,449 | 0.08427 | 10 | $\geq$ 3 |
| [Movielens-1M](#movielens) | 6,033 | 3,123 | 834,449 | 665,133 | 169,316 | 0.04429 | 10 | $\geq$ 3 |
| [Movielens-10M](#movielens) | 69,584 | 9,175 | 8,233,567 | 6,559,227 | 1,674,340 | 0.01290 | 10 | $\geq$ 3 |
| [Movielens-20M](#movielens) | 137,523 | 14,258 | 16,447,894 | 13,103,591 | 3,344,303 | 0.00839 | 10 | $\geq$ 3 |
| [Movielens-25M](#movielens) | 161,585 | 21,039 | 20,427,563 | 16,277,892 | 4,149,671 | 0.00601 | 10 | $\geq$ 3 |
| [Yahoo Music](#yahoo-music) | 15,400 | 1,000 | 365,704 | 286,540 | 79,164 | 0.02375 | 10 | $-$ |
| [Yelp2018](#yelp) | 55,616 | 34,945 | 1,506,777 | 1,183,556 | 323,221 | 0.00078 | 10 | $\geq$ 3 |
| [Yelp2022](#yelp) | 69,090 | 41,814 | 1,897,447 | 1,490,688 | 406,759 | 0.00066 | 10 | $\geq$ 3 |


------

### Amazon

#### Description

The Amazon dataset is a large crawl of product reviews from Amazon, including reviews, metadata and graphs. The latest Amazon dataset contains 233.1 million reviews in the range May 1996 - Oct 2018. Many previous research papers use the old version, which contains 142.8 million reviews spanning May 1996 - July 2014. We process these two versions of the Amazon dataset and name them as Amazon2018 and Amazon2014, respectively. We are planning to process the Amazon2023 dataset with 571.5 million reviews in the range May 1996 - Sep 2023.

The Amazon dataset is often categorized into different domains, such as Amazon-Book, Amazon-Beauty, Amazon-Electronics, etc. Currently, we have processed the following datasets:

**Amazon2023**: 

**Amazon2018**: 
- [Amazon2018](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFilesSmall/all_csv_files.csv): All reviews with 233,055,327 reviews. 
- [Amazon2018-Book](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFilesSmall/Books.csv): Book reviews with 51,311,621 reviews.

**Amazon2014**:
- [Amazon2014](https://snap.stanford.edu/data/amazon/productGraph/item_dedup.csv): All reviews with 82,677,131 reviews.
- [Amazon2014-Beauty](https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Beauty.csv): Beauty reviews with 2,023,070 reviews.
- [Amazon2014-Book](https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Books.csv): Book reviews with 22,507,155 reviews.
- [Amazon2014-CD](https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_CDs_and_Vinyl.csv): CD and Vinyl reviews with 3,749,004 reviews.
- [Amazon2014-Electronic](https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Electronics.csv): Electronic reviews with 7,824,482 reviews.
- [Amazon2014-Health](https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Health_and_Personal_Care.csv): Health and Personal Care reviews with 2,982,326 reviews.
- [Amazon2014-Movie](https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Movies_and_TV.csv): Movie and TV reviews with 4,607,047 reviews.

For the complete list of Amazon datasets, you can refer to:
- [RSPD/Amazon2023](https://amazon-reviews-2023.github.io/).
- [RSPD/Amazon2018](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/).
- [RSPD/Amazon2014](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html).

#### Processing Method

We adopt the "Ratings only" version for recommendation task, which only includes the `(item, user, rating, timestamp)` tuples, while the `rating` is the number of stars (1-5, has some error 0 ratings). We omit the `timestamp`, drop the duplicate reviews (if two reviews have the same user and item) and low-rating reviews (if the review has less than 3 stars), then apply 10-core filterin and obtain the training and testing sets with a ratio of 8:2. The filtered dataset statistics are shown as follows:
- Amazon2018: 1,591,189 users, 3,481,925 items and 88,111,403 reviews.
- Amazon2018-Book: 274,926 users, 537,851 items and 14,847,692 reviews.
- Amazon2014: 718,190 users, 460,233 items and 16,971,561 reviews.
- Amazon2014-Beauty: 1,237 users, 719 items and 25,736 reviews.
- Amazon2014-Book: 135,109 users, 115,172 items and 4,042,382 reviews.
- Amazon2014-CD: 12,784 users, 13,874 items and 360,763 reviews.
- Amazon2014-Electronic: 13,455 users, 8,360 items and 234,521 reviews.
- Amazon2014-Health: 1,974 users, 1,200 items and 48,189 reviews.
- Amazon2014-Movie: 26,968 users, 18,563 items and 762,957 reviews.


#### Final Dataset

| Amazon2018 | All Statistics | Book Statistics |
| ---------- | --------------- | --------------- |
| Users | 1,591,189 | 274,926 |
| Items | 3,481,925 | 537,851 |
| Interactions | 88,111,403 | 14,847,692 |
| Train Interactions | 69,856,556 | 11,768,846 |
| Test Interactions | 18,254,847 | 3,078,846 |
| Density | 0.00002 | 0.00010 |
| $N$-core | 10 | 10 |
| Rating | $\geq$ 3 | $\geq$ 3 |

| Amazon2014 | All Statistics | Beauty Statistics | Book Statistics | CD Statistics | Electronic Statistics | Health Statistics | Movie Statistics |
| ---------- | --------------- | --------------- | --------------- | --------------- | --------------- | --------------- | --------------- |
| Users | 718,190 | 1,237 | 135,109 | 12,784 | 13,455 | 1,974 | 26,968 |
| Items | 460,233 | 719 | 115,172 | 13,874 | 8,360 | 1,200 | 18,563 |
| Interactions | 16,971,561 | 25,736 | 4,042,382 | 360,763 | 234,521 | 48,189 | 762,957 |
| Train Interactions | 13,294,890 | 20,101 | 3,180,699 | 283,608 | 182,398 | 37,784 | 599,903 |
| Test Interactions | 3,676,671 | 5,635 | 861,683 | 77,155 | 52,123 | 10,405 | 163,054 |
| Density | 0.00005 | 0.02894 | 0.00026 | 0.00203 | 0.00208 | 0.02034 | 0.00152 |
| $N$-core | 10 | 10 | 10 | 10 | 10 | 10 | 10 |
| Rating | $\geq$ 3 | $\geq$ 3 | $\geq$ 3 | $\geq$ 3 | $\geq$ 3 | $\geq$ 3 | $\geq$ 3 |


#### Source

We collect the Amazon dataset from [RSPD/Amazon Product Reviews](https://cseweb.ucsd.edu/~jmcauley/datasets.html#amazon_reviews)'s "Ratings only" datasets: 
- [Amazon2023](https://amazon-reviews-2023.github.io/): The complete review data of Amazon2023 is in [Amazon2023-All](https://amazon-reviews-2023.github.io/data_processing/0core.html).
- [Amazon2018](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/): The complete review data of Amazon2018 is in [Amazon2018-All](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFilesSmall/all_csv_files.csv). 
- [Amazon2014](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html): The complete review data of Amazon2014 is in [Amazon2014-All](https://snap.stanford.edu/data/amazon/productGraph/item_dedup.csv).

The website also provides the raw review data, product metadata and 5-core datasets. 

You may also access the original papers: 
- He, Ruining, and Julian McAuley. "[Ups and downs: Modeling the visual evolution of fashion trends with one-class collaborative filtering.](https://dl.acm.org/doi/abs/10.1145/2872427.2883037)" proceedings of the 25th international conference on world wide web. 2016.
- McAuley, Julian, et al. "[Image-based recommendations on styles and substitutes.](https://dl.acm.org/doi/abs/10.1145/2766462.2767755)" Proceedings of the 38th international ACM SIGIR conference on research and development in information retrieval. 2015.

Our aim in the future is to complete the processing of the Amazon dataset and test all recommendation models on these datasets.

------

### Douban

#### Description

The Douban dataset is crawled from a popular Chinese social networking service platform [Douban](http://www.douban.com/), including reviews and ratings in three domains (i.e. movie, book and music): 

| Dataset | Users | Items | Interactions |
| ------- | ----- | ----- | ------------ |
| Douban-Movie | 94,890 | 81,906 | 11,742,260 |
| Douban-Book | 46,548 | 212,995 | 1,908,081 |
| Douban-Music | 39,742 | 164,223 | 1,792,501 |

#### Processing Method

The rating data is in the format of `(user, item, rating, timestamp)` tuples, e.g. 

| User | Item | Rating | Timestamp |
| ---- | ---- | ------ | --------- |
| 630157 | 0 | 5 | 1182009600.0 |
| 630157 | 1 | 5 | 1182009600.0 |
| 630157 | 2 | 4 | 1182009600.0 |
| 630157 | 3 | 5 | 1182355200.0 |
| 630157 | 4 | 5 | 1182355200.0 |

We omit the `timestamp`, drop the duplicate reviews (if two reviews have the same user and item) and low-rating reviews (if the review has less than 3 stars, including blank rating/-1), then apply 10-core filtering and obtain the training and testing sets with a ratio of 8:2.

#### Final Dataset

| Douban | Movie Statistics | Book Statistics | Music Statistics |
| ------ | ---------------- | ---------------- | --------------- |
| Users | 51,803 | 18,662 | 13,343 |
| Items | 25,993 | 15,207 | 15,957 |
| Interactions | 8,355,127 | 938,173 | 903,178 |
| Train Interactions | 6,664,958 | 743,085 | 717,188 |
| Test Interactions | 1,690,169 | 195,088 | 185,990 |
| Density | 0.00620 | 0.00331 | 0.00424 |
| $N$-core | 10 | 10 | 10 |
| Rating | $\geq$ 3 | $\geq$ 3 | $\geq$ 3 |


#### Source

We collect the Douban dataset from the implementation of original paper: [DGRec-Tensorflow](https://github.com/DeepGraphLearning/RecommenderSystems/tree/master/socialRec) and [DGRec-PyTorch](https://github.com/ssu-dmlab/DGRec-pytorch). The raw data can be downloaded from [Douban.tar.gz](https://www.dropbox.com/s/u2ejjezjk08lz1o/Douban.tar.gz?dl=0).

You may also access the original paper:
- Song, Weiping, et al. "[Session-based social recommendation via dynamic graph attention networks.](https://dl.acm.org/doi/abs/10.1145/3289600.3290989)" Proceedings of the Twelfth ACM international conference on web search and data mining. 2019.


------

### Food

#### Description

The Food dataset consists of 180K+ recipes and 700K+ recipe reviews covering 18 years of user interactions and uploads on Food.com (formerly GeniusKitchen). 

#### Processing Method

The dataset has three splits: `interactions_train.csv`, `interactions_validation.csv` and `interactions_test.csv` (however, we re-split the dataset), each row in the file is in the format of `(user_id, recipe_id, date, rating, u, i)`, e.g.

| user_id | recipe_id | date | rating | u | i |
| ------- | --------- | ---- | ------ | - | - |
| 2046 | 4684 | 2000-02-25 | 5.0 | 22095 | 44367 |
| 2046 | 517 | 2000-02-25 | 5.0 | 22095 | 87844 |
| 1773 | 7435 | 2000-03-13 | 5.0 | 24732 | 138181 |
| 1773 | 278 | 2000-03-13 | 4.0 | 24732 | 93054 |
| 2046 | 3431 | 2000-04-07 | 5.0 | 22095 | 101723 |

The `user_id` and `u`, `recipe_id` and `i` are one-to-one mapping, we can treat the `user_id` as the user ID, and the `recipe_id` as the item ID. We omit the `date`, drop the duplicate reviews (if two reviews have the same user and item) and low-rating reviews (if the review has less than 3 stars), then apply 10-core filtering and obtain the training and testing sets with a ratio of 8:2.

#### Final Dataset

| Food | Statistics |
| ---- | ---------- |
| Users | 5,875 |
| Items | 9,852 |
| Interactions | 233,038 |
| Train Interactions | 184,106 |
| Test Interactions | 48,932 |
| Density | 0.00403 |
| $N$-core | 10 |
| Rating | $\geq$ 3 |

#### Source

We collect the Food dataset from [Kaggle/Food](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions).


------

### Gowalla

#### Description

The Gowalla dataset is a check-in dataset collected from the location-based social network [Gowalla](https://en.wikipedia.org/wiki/Gowalla), including 107,092 users, 1,280,969 locations and 6,442,892 check-ins. 

#### Processing Method

We only adopt the check-in data in `loc-gowalla_totalCheckins.txt.gz`. The original check-in data format is shown as follows:

| user | check-in time | latitude | longitude | location id |
| ---- | -------------- | -------- | --------- | ----------- |
| 0 | 2010-10-19T23:55:27Z | 30.2359091167 | -97.7951395833 | 22847 | 
| 0 | 2010-10-18T22:17:43Z | 30.2691029532 | -97.7493953705 | 420315 | 
| 0 | 2010-10-17T23:42:03Z | 30.2557309927 | -97.7633857727 | 316637 | 
| 0 | 2010-10-17T19:26:05Z | 30.2634181234 | -97.7575966669 | 16516 | 
| 0 | 2010-10-16T18:50:42Z | 30.2742918584 | -97.7405226231 | 5535878| 

Let's focus on the `user` and `location id` columns. We can treat the `user` as the user ID, and the `location id` as the item ID. After removing the duplicate check-ins (if two check-ins have the same user and location), we obtain a dataset with 107,092 users, 1,280,969 locations and 3,981,334 check-ins. Applying 10-core filtering results in a dataset with 29,858 users, 40,988 locations and 1,027,464 check-ins. We split the dataset into training and testing sets with a ratio of 8:2.

#### Final Dataset

| Gowalla | Statistics |
| ----- | ---------- |
| Users | 29,858 |
| Items | 40,988 |
| Interactions | 1,027,464 |
| Train Interactions | 810,128 |
| Test Interactions | 217,336 |
| Density | 0.00084 |
| $N$-core | 10 |
| Rating | $-$ |

#### Source

We collect this dataset from [SNAP/Gowalla](https://snap.stanford.edu/data/loc-gowalla.html). 

You may also access the original paper: 
- Cho, Eunjoon, Seth A. Myers, and Jure Leskovec. "[Friendship and mobility: user movement in location-based social networks.](https://dl.acm.org/doi/abs/10.1145/2020408.2020579)" Proceedings of the 17th ACM SIGKDD international conference on Knowledge discovery and data mining. 2011.

------

### Movielens

#### Description

The Movielens dataset is a rating data set from the [MovieLens](https://movielens.org). Currently, it provides several versions of datasets and we process the following versions:
- [Movielens-100k](https://files.grouplens.org/datasets/movielens/ml-100k.zip): MovieLens 100K movie ratings. Stable benchmark dataset. 100,000 ratings from 1000 users on 1700 movies. Released 4/1998.
- [Movielens-1M](https://files.grouplens.org/datasets/movielens/ml-1m.zip): MovieLens 1M movie ratings. Stable benchmark dataset. 1 million ratings from 6000 users on 4000 movies. Released 2/2003.
- [MovieLens 10M Dataset](https://files.grouplens.org/datasets/movielens/ml-10m.zip): MovieLens 10M movie ratings. Stable benchmark dataset. 10 million ratings and 100,000 tag applications applied to 10,000 movies by 72,000 users. Released 1/2009.
- [MovieLens 20M Dataset](https://files.grouplens.org/datasets/movielens/ml-20m.zip): MovieLens 20M movie ratings. Stable benchmark dataset. 20 million ratings and 465,000 tag applications applied to 27,000 movies by 138,000 users. Includes tag genome data with 12 million relevance scores across 1,100 tags. Released 4/2015; updated 10/2016 to update links.csv and add tag genome data.
- [MovieLens 25M Dataset](https://files.grouplens.org/datasets/movielens/ml-25m.zip): MovieLens 25M movie ratings. Stable benchmark dataset. 25 million ratings and one million tag applications applied to 62,000 movies by 162,000 users. Includes tag genome data with 15 million relevance scores across 1,129 tags. Released 12/2019.

#### Processing Method

Despite the different versions and file names, the rating data is in the same format, i.e. `(user, item, rating, timestamp)` tuples, e.g. 

| userId | movieId | rating | timestamp |
| ------ | ------- | ------ | --------- |
| 1 | 296 | 5.0 | 1147880044 |
| 1 | 306 | 3.5 | 1147868817 |
| 1 | 307 | 5.0 | 1147868828 |
| 1 | 665 | 5.0 | 1147878820 |
| 1 | 899 | 3.5 | 1147868510 |

We omit the `timestamp`, drop the duplicate ratings (if two ratings have the same user and item) and low-rating ratings (if the rating has less than 3 stars, note that there may be half-star increments), then apply 10-core filtering and obtain the training and testing sets with a ratio of 8:2.

#### Final Dataset

| Movielens | 100k Statistics | 1M Statistics | 10M Statistics | 20M Statistics | 25M Statistics |
| --------- | --------------- | ------------- | -------------- | -------------- | -------------- |
| Users | 939 | 6,033 | 69,584 | 137,523 | 161,585 |
| Items | 1,016 | 3,123 | 9,175 | 14,258 | 21,039 |
| Interactions | 80,393 | 834,449| 8,233,567 | 16,447,894 | 20,427,563 |
| Train Interactions | 63,944 | 665,133 | 6,559,227 | 13,103,591 | 16,277,892 |
| Test Interactions | 16,449 | 169,316 | 1,674,340 | 3,344,303 | 4,149,671 |
| Density | 0.08427 | 0.04429 | 0.01290 | 0.00839 | 0.00601 |
| $N$-core | 10 | 10 | 10 | 10 | 10 |
| Rating | $\geq$ 3 | $\geq$ 3 | $\geq$ 3 | $\geq$ 3 | $\geq$ 3 |

#### Source

We collect the Movielens dataset from [GroupLens/MovieLens](https://grouplens.org/datasets/movielens/).

You may also access the original paper:
- F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19 (December 2015), 19 pages. DOI=http://dx.doi.org/10.1145/2827872

------

### Yahoo Music

#### Description

The Yahoo Music dataset represents a snapshot of the Yahoo! Music community's preferences for various musical artists. The dataset contains over ten million ratings of musical artists given by Yahoo! Music users over the course of a one month period sometime prior to March 2004. 

You may request the dataset from the Yahoo! Webscope program. The dataset is available in the [R1 - Yahoo! Music User Ratings of Musical Artists, version 1.0 (423 MB)](https://webscope.sandbox.yahoo.com/catalog.php?datatype=r). However, currently we do not get the raw version of the dataset, and we will adopt the processed dataset in the paper [Distributionally Robust Graph-based Recommendation System](https://arxiv.org/abs/2402.12994). 

#### Processing Method

The processed dataset we use has already satisfied the 10-core filtering, thus our data-clean process does not discard any interactions. We only re-shuffle and split the dataset into training and testing sets with a ratio of 8:2. Note that we do not filter the low-rating reviews in this dataset.

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

#### Source

We collect the Yahoo Music dataset from the implementation of the paper: [Distributionally Robust Graph-based Recommendation System](https://arxiv.org/abs/2402.12994).

The original dataset can be accessed from the Yahoo! Webscope program: [R1 - Yahoo! Music User Ratings of Musical Artists, version 1.0 (423 MB)](https://webscope.sandbox.yahoo.com/catalog.php?datatype=r).

------

### Yelp

#### Description

The Yelp dataset is a subset of Yelp's businesses, reviews, and user data. It was originally put together for the Yelp Dataset Challenge which is a chance for students to conduct research or analysis on Yelp's data and share their discoveries. Our recommendation task focuses on the review data. 

Here we process two versions of the Yelp dataset: Yelp2018 and Yelp2022, with 5,261,669 and 6,990,280 reviews, respectively.


#### Processing Method

Yelp dataset including five files: `business.json`, `review.json`, `user.json`, `checkin.json`, and `tip.json`. Here we only use the `review.json` file. For more details about the original Yelp dataset format, you can refer to the [Yelp Dataset JSON](https://www.yelp.com/dataset/documentation/main). 

`review.json` contains full review text data including the `user_id` that wrote the review and the `business_id` the review is written for. Here is an example of the original review data:

```json
{
    // string, 22 character unique review id
    "review_id": "zdSx_SD6obEhz9VrW9uAWA",
    // string, 22 character unique user id, maps to the user in user.json
    "user_id": "Ha3iJu77CxlrFm-vQRs_8g",
    // string, 22 character business id, maps to business in business.json
    "business_id": "tnhfDv5Il8EaGSXZGiuQGg",
    // integer, star rating
    "stars": 4,
    // string, date formatted YYYY-MM-DD
    "date": "2016-03-09",
    // string, the review itself
    "text": "Great place to hang out after work: the prices are decent, and the ambience is fun. It's a bit loud, but very lively. The staff is friendly, and the food is good. They have a good selection of drinks.",
    // integer, number of useful votes received
    "useful": 0,
    // integer, number of funny votes received
    "funny": 0,
    // integer, number of cool votes received
    "cool": 0
}
```

Here we only use the `user_id`, `business_id` and `stars` (1-5) columns. We treat the `user_id` as the user ID, and the `business_id` as the item ID. After removing the duplicate reviews (if two reviews have the same user and business) and low-star reviews (if the review has less than 3 stars), we obtain a Yelp2018 dataset with 1,077,637 users, 169,895 businesses and 4,092,144 reviews, and a Yelp2022 dataset with 1,536,687 users, 148,556 businesses and 5,218,826 reviews. These reviews are considered as successful deals and positive interactions. Applying 10-core filtering results in a Yelp2018 dataset with 55,616 users, 34,945 businesses and 1,506,777 reviews, and a Yelp2022 dataset with 69,090 users, 41,814 businesses and 1,897,447 reviews. We split the dataset into training and testing sets with a ratio of 8:2.

#### Final Dataset

| Yelp | Yelp2018 Statistics | Yelp2022 Statistics |
| ----- | ---------- | ---------- |
| Users | 55,616 | 69,090 |
| Items | 34,945 | 41,814 |
| Interactions | 1,506,777 | 1,897,447 |
| Train Interactions | 1,183,556 | 1,490,688 |
| Test Interactions | 323,221 | 406,759 |
| Density | 0.00078 | 0.00066 |
| $N$-core | 10 | 10 |
| Rating | $\geq$ 3 | $\geq$ 3 |

#### Source

We collect the [Yelp2018](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset/versions/1) and [Yelp2022](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset/versions/4) datasets from Kaggle, which are the first and latest versions of the Yelp Dataset. You may also access the latest version of the dataset in [Yelp Open Dataset](https://www.yelp.com/dataset). We may also plan to collect the other versions of Yelp datasets in the future.

Most of the research papers use the Yelp2018 dataset (first used in Yelp Challenge 2018). However, we hope the researchers can also consider testing their methods on the latest Yelp2022 dataset to avoid overfitting to the Yelp2018 dataset, since the (star and time) distribution of the Yelp2022 dataset is very different from the Yelp2018 dataset. 



------

## Contributing New Datasets

We sincerely welcome the researchers to contribute new datasets to our IR Benchmark. You should satisfy the following requirements:
- The dataset should in the format of what we have described in [Dataset Format and Cleaning](#dataset-format-and-cleaning). The five files `user_list.tsv`, `item_list.tsv`, `train.tsv`, `test.tsv` and `summary.txt` should be provided.
- The original dataset is not required to be public or uploaded to our repository. However, you should provide the processing method and update the `data/dataset_summary.md` (i.e. this file) to describe the dataset and its statistics.
- After satisfying the above requirements, you can submit a pull request to our repository. We will review it and merge it into our repository if it meets our requirements.

Finally, we thanks for your kind contribution to our IR Benchmark. If it's possible, we will test the current recommendation models on your dataset and update the benchmark results.



------

## Useful Links

- [RSPD](https://cseweb.ucsd.edu/~jmcauley/datasets.html): Recommender Systems and Personalization Datasets, a collection of famous datasets for recommendation systems used in Julian McAuley's lab, UCSD.
- [SNAP](https://snap.stanford.edu/): Stanford Network Analysis Project, a general purpose network analysis and graph mining library. 
- [RecBole](https://github.com/RUCAIBox/RecBole): A unified and comprehensive recommendation library, including various recommendation models and datasets. This library collects many recommendation datasets and can be found in [RecBole/DatasetList](https://recbole.io/dataset_list.html) and [RecSysDatasets](https://github.com/RUCAIBox/RecSysDatasets). 
- [RecZoo](https://huggingface.co/reczoo#datasets): A collection of recommendation datasets in Hugging Face.
- [RSTasks](https://paperswithcode.com/task/recommendation-systems): A collection of recommendation system benchmarks in Papers with Code.


