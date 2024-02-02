# Item Recommendation Benchmark


## :memo: Introduction

This repository collects the open source **Item Recommendation benchmark datasets and SOTA models**, in order to facilitate the research of item recommendation. 

In this repository, we provide the unified interface for data processing, model selection, training and evaluation. We hope that this repository can help researchers to quickly reproduce the SOTA models and fairly compare their preformance. 

Our aim is to build a up-to-date benchmark for Item Recommendation. We will be very grateful if you want to contribute new datasets or models to this repository.

This code is based on [NNI (Neural Network Intelligence)](https://github.com/microsoft/nni) framework, a famous AutoML toolkit developed by Microsoft. We use NNI to manage the hyperparameter optimization and training process. 


## :chart_with_upwards_trend: Tasks & Evaluation Metrics

In this section, we will briefly introduce the tasks and evaluation metrics in Item Recommendation.





## :computer: CLI Usage




## :package: Benchmark Datasets



## :1st_place_medal: SOTA Models





------

## :thought_balloon: Feedback

This repository is initially built by [Tiny Snow](https://github.com/TinySnow) for research purpose. He is a master student at [College of Computer Science and Technology, Zhejiang University](http://www.cs.zju.edu.cn/) supervised by [Prof. Jiawei Chen](https://jiawei-chen.github.io/). He is also a member of ZLST Lab led by [Prof. Chun Chen](https://mypage.zju.edu.cn/chenc), Member of [CAE](https://www.cae.cn/). 

If you want to contribute to this repository, please feel free to pull requests. 
- **Contribute dataset**: your dataset should be added to `data/` directory. You need to provide three standard files: `train.txt`, `test.txt` and `data_summary.txt`. See `data/dataset_format.md` for more details.
- **Contribute model**: your model should be compatible with the unified interface, i.e. inheriting the standard base classes and implementing the arg parser. You should at least provide a `README.md` file to briefly introduce your model, running instructions, arugment yaml file and the performance on current benchmark datasets. See `models/model_format.md` for more details.

If you have any questions, please open an issue or contact [Tiny Snow](https://github.com/TinySnow): 
- Email: [futurelover10032@gmail.com](mailto:futurelover10032@gmail.com)
- Google Scholar: [Tiny Snow](https://scholar.google.com/citations?hl=en&user=ajZIwAgAAAAJ)


## :balance_scale: License

This software is provided under the [GPL-3.0 license](https://choosealicense.com/licenses/gpl-3.0/) © 2024 Tiny Snow. All rights reserved.

