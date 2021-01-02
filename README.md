# CRSLab

[![Pypi Latest Version](https://img.shields.io/pypi/v/crslab)](https://pypi.org/project/crslab)
[![Release](https://img.shields.io/github/v/release/rucaibox/crslab.svg)](https://github.com/rucaibox/crslab/releases)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)

[Paper]() | [中文版](./README_CN.md)

**CRSLab** is the **first** open-source toolkit for building Conversational Recommender System (CRS). It is developed based on Python and PyTorch. CRSLab has the following highlights:

- **Comprehensive benchmark models and datasets**: We have integrated commonly-used 6 datasets and 18 models, including KG-based and pre-training models such as GCN, BERT and GPT-2. We have preprocessed these datasets to support these models, and release for downloading.
- **Extensive and standard evaluation protocols**: We support a series of widely-adopted evaluation protocols for testing and comparing different CRS.
- **General and extensible structure**: We design a general and extensible structure to unify various conversational recommendation datasets and models, in which we integrate various built-in interfaces and functions for quickly development.
- **Easy to quick start**: We provide simple yet flexible configuration for new researchers to quick start integrated models in our library. 
- **Human-machine interaction interfaces**: We provide flexible human-machine interaction interfaces for researchers to do quantitive analysis.

<p align="center">
  <img src="https://i.loli.net/2020/12/30/6TPVG4pBg2rcDf9.png" alt="RecBole v0.1 architecture" width="400">
  <br>
  <b>Figure 1</b>: The overall framework of CRSLab
</p>




- [Installation](#Installation)
- [Quick-Start](#Quick-Start)
- [Models](#Models)
- [Datasets](#Datasets)
- [Performance](#Performance)
- [Releases](#Releases)
- [Contributions](#Contributions)
- [Citing](#Citing)
- [Team](#Team)
- [License](#License)



## Installation

CRSLab works with the following operating systems：

- Linux
- Windows 10
- macOS X

CRSLab requires Python version 3.6 or later.

CRSLab requires torch version 1.4.0 or later. If you want to use CRSLab with GPU, please ensure that CUDA or CUDAToolkit version is 9.2 or later. Please use the combinations shown in this [Link](https://pytorch-geometric.com/whl/) to ensure the normal operation of PyTorch Geometric.



### Install PyTorch

Use PyTorch [Locally Installation](https://pytorch.org/get-started/locally/) or [Previous Versions Installation](https://pytorch.org/get-started/previous-versions/) commands to install PyTorch. For example, on Linux and Windows 10:

```bash
# CUDA 10.1
pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

# CPU only
pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

If you want to use CRSLab with GPU, make sure the following command prints `True` after installation:

```bash
$ python -c "import torch; print(torch.cuda.is_available())"
>>> True
```



### Install PyTorch Geometric

Ensure that at least PyTorch 1.4.0 is installed:

```bash
$ python -c "import torch; print(torch.__version__)"
>>> 1.6.0
```

Find the CUDA version PyTorch was installed with:

```bash
$ python -c "import torch; print(torch.version.cuda)"
>>> 10.1
```

Install the relevant packages:

```bash
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric
```

where `${CUDA}` and `${TORCH}` should be replaced by your specific CUDA version (`cpu`, `cu92`, `cu101`, `cu102`, `cu110`) and PyTorch version (`1.4.0`, `1.5.0`, `1.6.0`, `1.7.0`) respectively. For example, for PyTorch 1.6.0 and CUDA 10.1, type:

```bash
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.6.0+cu101.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.6.0+cu101.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.6.0+cu101.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.6.0+cu101.html
pip install torch-geometric
```



### Install CRSLab

```bash
git clone https://github.com/RUCAIBox/CRSLab && cd CRSLab
pip install -e .
```



## Quick-Start

With the source code, you can use the provided script for initial usage of our library:

```bash
python run_crslab.py --config config/kgsf/redial.yaml
```

The system will complete the data preprocessing, and training, validation, testing of each model in turn. Finally it will get the evaluation results of specified models.

If you want to save pre-processed datasets and training results of models, you can use the following command:

```bash
python run_crslab.py --config config/kgsf/redial.yaml --save_data --save_system
```

In summary, there are following arguments in `run_crslab.py`:

- `--config` or `-c`: relative path for configuration file.
- `--save_data` or `-sd`: save pre-processed dataset.
- `--restore_data` or `-rd`: restore pre-processed dataset from file.
- `--save_system` or `-ss`: save trained system.
- `--restore_system` or `-rs`: restore trained system from file.
- `--debug` or `-d`: use valid dataset to debug your system.
- `--interact` or `-i`: interact with your system instead of training.



## Models

As the first release version, we have implemented 18 models in the four categories of CRS model, Recommendation model, Policy model and Conversation model.

|       Category       |                            Model                             |      Graph Neural Network?      |       Pre-training Model?       |
| :------------------: | :----------------------------------------------------------: | :-----------------------------: | :-----------------------------: |
|      CRS Model       | [ReDial](https://arxiv.org/abs/1812.07617)<br/>[KBRD](https://arxiv.org/abs/1908.05391)<br/>[KGSF](https://arxiv.org/abs/2007.04032)<br/>[TG-ReDial](https://arxiv.org/abs/2010.04125) |       ×<br/>√<br/>√<br/>×       |       ×<br/>×<br/>×<br/>√       |
| Recommendation model | Popularity<br/>[GRU4Rec](https://arxiv.org/abs/1609.05787)<br/>[SASRec](https://arxiv.org/abs/1808.09781)<br/>[TextCNN](https://arxiv.org/abs/1408.5882)<br/>[R-GCN](https://arxiv.org/abs/1703.06103)<br/>[BERT](https://arxiv.org/abs/1810.04805) | ×<br/>×<br/>×<br/>×<br/>√<br/>× | ×<br/>×<br/>×<br/>×<br/>×<br/>√ |
|     Policy model     | PMI<br/>[MGCG](https://arxiv.org/abs/2005.03954)<br/>[Conv-BERT](https://arxiv.org/abs/2010.04125)<br/>[Topic-BERT](https://arxiv.org/abs/2010.04125)<br/>[Profile-BERT](https://arxiv.org/abs/2010.04125) |    ×<br/>×<br/>×<br/>×<br/>×    |    ×<br/>×<br/>√<br/>√<br/>√    |
|  Conversation model  | [HERD](https://arxiv.org/abs/1507.04808)<br/>[Transformer](https://arxiv.org/abs/1706.03762)<br/>[GPT-2](http://www.persagen.com/files/misc/radford2019language.pdf) |          ×<br/>×<br/>×          |          ×<br/>×<br/>√          |

Among them, the four CRS models integrate the recommendation model and the conversation model to improve each other.

For Recommendation model and Conversation model, we have respectively implemented the following commonly-used automatic evaluation metrics:

|        Category        |                           Metrics                            |
| :--------------------: | :----------------------------------------------------------: |
| Recommendation Metrics |      Hit@{1, 10, 50}, MRR@{1, 10, 50}, NDCG@{1, 10, 50}      |
|  Conversation Metrics  | PPL, BLEU-{1, 2, 3, 4}, Embedding Average/Extreme/Greedy, Distinct-{1, 2, 3, 4} |



## Datasets

We have collected and preprocessed 6 commonly-used human-annotated datasets, and each dataset was matched with proper KGs as shown below:

|                           Dataset                            | Dialogs | Utterances |   Domains    | Task Definition | Entity KG  |  Word KG   |
| :----------------------------------------------------------: | :-----: | :--------: | :----------: | :-------------: | :--------: | :--------: |
|       [ReDial](https://redialdata.github.io/website/)        | 10,006  |  182,150   |    Movie     |       --        |  DBpedia   | ConceptNet |
|      [TG-ReDial](https://github.com/RUCAIBox/TG-ReDial)      | 10,000  |  129,392   |    Movie     |   Topic Guide   | CN-DBpedia |   HowNet   |
|        [GoRecDial](https://arxiv.org/abs/1909.03922)         |  9,125  |  170,904   |    Movie     |  Action Choice  |  DBpedia   | ConceptNet |
|        [DuRecDial](https://arxiv.org/abs/2005.03954)         | 10,200  |  156,000   | Movie, Music |    Goal Plan    | CN-DBpedia |   HowNet   |
|      [INSPIRED](https://github.com/sweetpeach/Inspired)      |  1,001  |   35,811   |    Movie     | Social Strategy |  DBpedia   | ConceptNet |
| [OpenDialKG](https://github.com/facebookresearch/opendialkg) | 13,802  |   91,209   | Movie, Book  |  Path Generate  |  DBpedia   | ConceptNet |



## Performance

We have trained our models on the TG-Redial datasets and recorded relevant evaluation results.



### CRS Model

Recommendation Task:

|   Model   |    Hit@1    | Hit@10  | Hit@50  |  MRR@1   | MRR@10  | MRR@50  |  NDCG@1  | NDCG@10 | NDCG@50 |
| :-------: | :---------: | :-----: | :-----: | :------: | :-----: | :-----: | :------: | :-----: | :-----: |
|   KBRD    |  0.004011   | 0.0254  | 0.05882 | 0.004011 | 0.00891 | 0.01028 | 0.004011 | 0.01271 | 0.01977 |
|   KGSF    |  0.005793   | 0.02897 | 0.08155 | 0.005793 | 0.01195 | 0.01433 | 0.005793 | 0.01591 | 0.02738 |
| TG-ReDial |  0.007926   | 0.0251  | 0.0524  | 0.007926 | 0.01223 | 0.01341 | 0.007926 | 0.01523 | 0.0211  |

Generation Task:

|   Model   | BLEU@1 | BLEU@2  |  BLEU@3  |  BLEU@4   | Dist@1 | Dist@2 | Dist@3 | Dist@4 | Average | Extreme | Greedy |  PPL  |
| :-------: | :----: | :-----: | :------: | :-------: | :----: | :----: | :----: | :----: | :-----: | :-----: | :----: | :---: |
|   KBRD    | 0.2672 | 0.04582 | 0.01338  | 0.005786  | 0.4690 | 1.504  | 3.397  | 4.899  | 0.8626  | 0.3982  | 0.7102 | 52.54 |
|   KGSF    |        |         |          |           |        |        |        |        |         |         |        |       |
| TG-ReDial | 0.1254 | 0.02035 | 0.003544 | 0.0008028 | 0.8809 | 1.746  | 6.997  | 11.99  | 0.8104  | 0.3315  | 0.5981 | 7.41  |

Policy Task:

|   Model   | Hit@1  | Hit@10 | Hit@50 | MRR@1  | MRR@10 | MRR@50 | NDCG@1 | NDCG@10 | NDCG@50 |
| :-------: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :-----: | :-----: |
| TG-ReDial | 0.6004 | 0.8296 | 0.8926 | 0.6004 | 0.6928 | 0.6960 | 0.6004 | 0.7268  | 0.7410  |



### Recommendation Model

|  Model  |   Hit@1   |  Hit@10  | Hit@50  |   MRR@1   |  MRR@10   | MRR@50  |  NDCG@1   | NDCG@10  | NDCG@50  |
| :-----: | :-------: | :------: | :-----: | :-------: | :-------: | :-----: | :-------: | :------: | :------: |
| SASRec  | 0.0004456 | 0.001337 | 0.01604 | 0.0004456 | 0.0005756 | 0.00114 | 0.0004456 | 0.000745 | 0.003802 |
| TextCNN | 0.002674  | 0.01025  | 0.02362 | 0.002674  | 0.004339  | 0.00493 | 0.002674  | 0.005704 | 0.008595 |
|  BERT   |     -     | 0.004902 | 0.02807 |  0.07219  |  0.0106   | 0.01241 | 0.004902  | 0.01465  |  0.0239  |



### Policy Model

|   Model    | Hit@1  | Hit@10 | Hit@50 | MRR@1  | MRR@10 | MRR@50 | NDCG@1 | NDCG@10 | NDCG@50 |
| :--------: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :-----: | :-----: |
|    MGCG    | 0.5914 | 0.8184 | 0.8826 | 0.5914 | 0.6799 | 0.6831 | 0.5914 | 0.7124  | 0.7286  |
| Conv-BERT  | 0.5974 | 0.8141 | 0.8809 | 0.5974 | 0.6840 | 0.6873 | 0.5974 | 0.7163  | 0.7312  |
| Topic-BERT | 0.5981 | 0.8278 | 0.8854 | 0.5981 | 0.6902 | 0.6930 | 0.5981 | 0.7243  | 0.7373  |



### Conversation Model

|    Model    | BLEU@1  | BLEU@2  |  BLEU@3  |  BLEU@4   | Dist@1 | Dist@2 | Dist@3  | Dist@4 | Average | Extreme | Greedy |  PPL  |
| :---------: | :-----: | :-----: | :------: | :-------: | :----: | :----: | :-----: | :----: | :-----: | :-----: | :----: | :---: |
|    GPT2     | 0.08583 | 0.01187 | 0.003765 |  0.01095  | 2.348  | 4.624  |  8.843  | 12.46  | 0.7628  | 0.2971  | 0.5828 | 9.257 |
| Transformer | 0.2662  | 0.04396 | 0.01448  | 0.006512  | 0.3236 | 0.8371 |  2.023  | 3.056  | 0.8786  | 0.4384  | 0.6801 | 30.91 |
|    HERD     | 0.1199  | 0.01408 | 0.001358 | 0.0003501 | 0.1810 | 0.3691 | 0.84681 | 1.298  | 0.6969  | 0.3816  | 0.6388 | 472.3 |





## Releases

| Releases |      Date      |   Features   |
| :------: | :------------: | :----------: |
|  v0.1.1  | 12 / 31 / 2020 | Basic CRSLab |



## Contributions

Please let us know if you encounter a bug or have any suggestions by [filing an issue](https://github.com/RUCAIBox/CRSLab/issues).

We welcome all contributions from bug fixes to new features and extensions.

We expect all contributions discussed in the issue tracker and going through PRs.



## Citing

If you find CRSLab useful for your research or development, please cite the following [Paper]():

```

```



## Team

**CRSLab** was developed and maintained by [AI Box](http://aibox.ruc.edu.cn/) group in RUC.



## License

**CRSLab** uses [MIT License](./LICENSE).

