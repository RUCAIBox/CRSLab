# CRSLab

[![Pypi Latest Version](https://img.shields.io/pypi/v/crslab)](https://pypi.org/project/crslab)
[![Release](https://img.shields.io/github/v/release/rucaibox/crslab.svg)](https://github.com/rucaibox/crslab/releases)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-CRSLab-%23B21B1B)](https://arxiv.org/abs/2101.00939)
[![Documentation Status](https://readthedocs.org/projects/crslab/badge/?version=latest)](https://crslab.readthedocs.io/en/latest/?badge=latest)

[论文](https://arxiv.org/pdf/2101.00939.pdf) | [文档](https://crslab.readthedocs.io/en/latest/?badge=latest)
| [English Version](./README.md)

**CRSLab** 是一个用于构建对话推荐系统（CRS）的开源工具包，其基于 PyTorch 实现、主要面向研究者使用，并具有如下特色：

- **全面的基准模型和数据集**：我们集成了常用的 6 个数据集和 18 个模型，包括基于图神经网络和预训练模型，比如  GCN，BERT 和 GPT-2；我们还对数据集进行相关处理以支持这些模型，并提供预处理后的版本供大家下载。
- **大规模的标准评测**：我们支持一系列被广泛认可的评估方式来测试和比较不同的 CRS。
- **通用和可扩展的结构**：我们设计了通用和可扩展的结构来统一各种对话推荐数据集和模型，并集成了多种内置接口和函数以便于快速开发。
- **便捷的使用方法**：我们为新手提供了简单而灵活的配置，方便其快速启动集成在 CRSLab 中的模型。
- **人性化的人机交互接口**：我们提供了人性化的人机交互界面，以供研究者对比和测试不同的模型系统。

<p align="center">
  <img src="https://i.loli.net/2020/12/30/6TPVG4pBg2rcDf9.png" alt="RecBole v0.1 architecture" width="400">
  <br>
  <b>图片</b>: CRSLab 的总体架构
</p>




- [安装](#安装)
- [快速上手](#快速上手)
- [模型](#模型)
- [数据集](#数据集)
- [评测结果](#评测结果)
- [发行版本](#发行版本)
- [贡献](#贡献)
- [引用](#引用)
- [项目团队](#项目团队)
- [免责声明](#免责声明)



## 安装

CRSLab 可以在以下几种系统上运行：

- Linux
- Windows 10
- macOS X

CRSLab 需要在 Python 3.6 或更高的环境下运行。

CRSLab 要求 torch 版本在 1.4.0 及以上，如果你想在 GPU 上运行 CRSLab，请确保你的 CUDA 版本或者 CUDAToolkit 版本在 9.2 及以上。为保证 PyTorch Geometric 库的正常运行，请使用[链接](https://pytorch-geometric.com/whl/)所示的安装方式。



### 安装 PyTorch

使用 PyTorch [本地安装](https://pytorch.org/get-started/locally/)命令或者[先前版本安装](https://pytorch.org/get-started/previous-versions/)命令安装 PyTorch，比如在 Linux 和 Windows 下：

```bash
# CUDA 10.1
pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

# CPU only
pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

安装完成后，如果你想在 GPU 上运行 CRSLab，请确保如下命令输出`True`：

```bash
$ python -c "import torch; print(torch.cuda.is_available())"
>>> True
```



### 安装 PyTorch Geometric

确保安装的 PyTorch 版本至少为 1.4.0：

```bash
$ python -c "import torch; print(torch.__version__)"
>>> 1.6.0
```

找到安装好的 PyTorch 对应的 CUDA 版本：

```bash
$ python -c "import torch; print(torch.version.cuda)"
>>> 10.1
```

安装相关的包：

```bash
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric
```

其中`${CUDA}`和`${TORCH}`应使用确定的 CUDA 版本（`cpu`，`cu92`，`cu101`，`cu102`，`cu110`）和 PyTorch 版本（`1.4.0`，`1.5.0`，`1.6.0`，`1.7.0`）来分别替换。比如，对于 PyTorch 1.6.0 和 CUDA 10.1，输入：

```bash
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.6.0+cu101.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.6.0+cu101.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.6.0+cu101.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.6.0+cu101.html
pip install torch-geometric
```



### 安装 CRSLab

你可以通过 pip 来安装：

```bash
pip install crslab
```

也可以通过源文件进行进行安装：

```bash
git clone https://github.com/RUCAIBox/CRSLab && cd CRSLab
pip install -e .
```



## 快速上手

从 GitHub 下载 CRSLab 后，可以使用提供的脚本快速运行和测试，默认使用CPU：

```bash
python run_crslab.py --config config/crs/kgsf/redial.yaml
```

系统将依次完成数据的预处理，以及各模块的训练、验证和测试，并得到指定的模型评测结果。

如果你希望保存数据预处理结果与模型训练结果，可以使用如下命令：

```bash
python run_crslab.py --config config/crs/kgsf/redial.yaml --save_data --save_system
```

总的来说，`run_crslab.py`有如下参数可供调用：

- `--config` 或 `-c`：配置文件的相对路径，以指定运行的模型与数据集。
- `--gpu` or `-g`：指定 GPU id，支持多 GPU，默认使用 CPU（-1）。
- `--save_data` 或 `-sd`：保存预处理的数据。
- `--restore_data` 或 `-rd`：从文件读取预处理的数据。
- `--save_system` 或 `-ss`：保存训练好的 CRS 系统。
- `--restore_system` 或 `-rs`：从文件载入提前训练好的系统。
- `--debug` 或 `-d`：用验证集代替训练集以方便调试。
- `--interact` 或 `-i`：与你的系统进行对话交互，而非进行训练。
- `--tensorboard` or `-tb`：使用 tensorboardX 组件来监测训练表现。



## 模型

在第一个发行版中，我们实现了 4 类共 18 个模型。这里我们将对话推荐任务主要拆分成三个任务：推荐任务（生成推荐的商品），对话任务（生成对话的回复）和策略任务（规划对话推荐的策略）。其中所有的对话推荐系统都具有对话和推荐任务，他们是对话推荐系统的核心功能。而策略任务是一个辅助任务，其致力于更好的控制对话推荐系统，在不同的模型中的实现也可能不同（如 TG-ReDial 采用一个主题预测模型，DuRecDial 中采用一个对话规划模型等）：



|   类别   |                             模型                             |      Graph Neural Network?      |       Pre-training Model?       |
| :------: | :----------------------------------------------------------: | :-----------------------------: | :-----------------------------: |
| CRS 模型 | [ReDial](https://arxiv.org/abs/1812.07617)<br/>[KBRD](https://arxiv.org/abs/1908.05391)<br/>[KGSF](https://arxiv.org/abs/2007.04032)<br/>[TG-ReDial](https://arxiv.org/abs/2010.04125)<br/>[INSPIRED](https://www.aclweb.org/anthology/2020.emnlp-main.654.pdf) |    ×<br/>√<br/>√<br/>×<br/>×    |    ×<br/>×<br/>×<br/>√<br/>√    |
| 推荐模型 | Popularity<br/>[GRU4Rec](https://arxiv.org/abs/1511.06939)<br/>[SASRec](https://arxiv.org/abs/1808.09781)<br/>[TextCNN](https://arxiv.org/abs/1408.5882)<br/>[R-GCN](https://arxiv.org/abs/1703.06103)<br/>[BERT](https://arxiv.org/abs/1810.04805) | ×<br/>×<br/>×<br/>×<br/>√<br/>× | ×<br/>×<br/>×<br/>×<br/>×<br/>√ |
| 对话模型 | [HERD](https://arxiv.org/abs/1507.04808)<br/>[Transformer](https://arxiv.org/abs/1706.03762)<br/>[GPT-2](http://www.persagen.com/files/misc/radford2019language.pdf) |          ×<br/>×<br/>×          |          ×<br/>×<br/>√          |
| 策略模型 | PMI<br/>[MGCG](https://arxiv.org/abs/2005.03954)<br/>[Conv-BERT](https://arxiv.org/abs/2010.04125)<br/>[Topic-BERT](https://arxiv.org/abs/2010.04125)<br/>[Profile-BERT](https://arxiv.org/abs/2010.04125) |    ×<br/>×<br/>×<br/>×<br/>×    |    ×<br/>×<br/>√<br/>√<br/>√    |


其中，CRS 模型是指直接融合推荐模型和对话模型，以相互增强彼此的效果，故其内部往往已经包含了推荐、对话和策略模型。其他如推荐模型、对话模型、策略模型往往只关注以上任务中的某一个。

我们对于这几类模型，我们还分别实现了如下的自动评测指标模块：

|   类别   |                             指标                             |
| :------: | :----------------------------------------------------------: |
| 推荐指标 |      Hit@{1, 10, 50}, MRR@{1, 10, 50}, NDCG@{1, 10, 50}      |
| 对话指标 | PPL, BLEU-{1, 2, 3, 4}, Embedding Average/Extreme/Greedy, Distinct-{1, 2, 3, 4} |
| 策略指标 | Accuracy, Hit@{1,3,5} |





## 数据集

我们收集了 6 个常用的人工标注数据集，并对它们进行了预处理（包括引入外部知识图谱），以融入统一的 CRS 任务中。如下为相关数据集的统计数据：

|                           Dataset                            | Dialogs | Utterances |   Domains    | Task Definition | Entity KG  |  Word KG   |
| :----------------------------------------------------------: | :-----: | :--------: | :----------: | :-------------: | :--------: | :--------: |
|       [ReDial](https://redialdata.github.io/website/)        | 10,006  |  182,150   |    Movie     |       --        |  DBpedia   | ConceptNet |
|      [TG-ReDial](https://github.com/RUCAIBox/TG-ReDial)      | 10,000  |  129,392   |    Movie     |   Topic Guide   | CN-DBpedia |   HowNet   |
|        [GoRecDial](https://arxiv.org/abs/1909.03922)         |  9,125  |  170,904   |    Movie     |  Action Choice  |  DBpedia   | ConceptNet |
|        [DuRecDial](https://arxiv.org/abs/2005.03954)         | 10,200  |  156,000   | Movie, Music |    Goal Plan    | CN-DBpedia |   HowNet   |
|      [INSPIRED](https://github.com/sweetpeach/Inspired)      |  1,001  |   35,811   |    Movie     | Social Strategy |  DBpedia   | ConceptNet |
| [OpenDialKG](https://github.com/facebookresearch/opendialkg) | 13,802  |   91,209   | Movie, Book  |  Path Generate  |  DBpedia   | ConceptNet |



## 评测结果

我们在 TG-ReDial 数据集上对模型进行了训练和测试，这里我们将数据集按照 8:1:1 切分。其中对于每条数据，我们从对话的第一轮开始，一轮一轮的进行推荐、策略生成、回复生成任务。下表记录了相关的评测结果。

### 推荐任务

|   模型    |    Hit@1    |   Hit@10   |   Hit@50   |    MRR@1    |   MRR@10   |   MRR@50   |   NDCG@1    |  NDCG@10   |  NDCG@50   |
| :-------: | :---------: | :--------: | :--------: | :---------: | :--------: | :--------: | :---------: | :--------: | :--------: |
|  SASRec   |  0.000446   |  0.00134   |   0.0160   |  0.000446   |  0.000576  |  0.00114   |  0.000445   |  0.00075   |  0.00380   |
|  TextCNN  |   0.00267   |   0.0103   |   0.0236   |   0.00267   |  0.00434   |  0.00493   |   0.00267   |  0.00570   |  0.00860   |
|   BERT    |   0.00722   |  0.00490   |   0.0281   |   0.00722   |   0.0106   |   0.0124   |   0.00490   |   0.0147   |   0.0239   |
|   KBRD    |   0.00401   |   0.0254   |   0.0588   |   0.00401   |  0.00891   |   0.0103   |   0.00401   |   0.0127   |   0.0198   |
|   KGSF    |   0.00535   | **0.0285** | **0.0771** |   0.00535   |   0.0114   | **0.0135** |   0.00535   | **0.0154** | **0.0259** |
| TG-ReDial | **0.00793** |   0.0251   |   0.0524   | **0.00793** | **0.0122** |   0.0134   | **0.00793** |   0.0152   |   0.0211   |



### 对话任务

|    模型     |  BLEU@1   |  BLEU@2   |   BLEU@3   |   BLEU@4   |  Dist@1  |  Dist@2  |  Dist@3  |  Dist@4  |  Average  |  Extreme  |  Greedy   |   PPL    |
| :---------: | :-------: | :-------: | :--------: | :--------: | :------: | :------: | :------: | :------: | :-------: | :-------: | :-------: | :------: |
|    HERD     |   0.120   |  0.0141   |  0.00136   |  0.000350  |  0.181   |  0.369   |  0.847   |   1.30   |   0.697   |   0.382   |   0.639   |   472    |
| Transformer |   0.266   |  0.0440   |   0.0145   |  0.00651   |  0.324   |  0.837   |   2.02   |   3.06   |   0.879   |   0.438   |   0.680   |   30.9   |
|    GPT2     |  0.0858   |  0.0119   |  0.00377   |   0.0110   | **2.35** | **4.62** | **8.84** | **12.5** |   0.763   |   0.297   |   0.583   |   9.26   |
|    KBRD     |   0.267   |  0.0458   |   0.0134   |  0.00579   |  0.469   |   1.50   |   3.40   |   4.90   |   0.863   |   0.398   |   0.710   |   52.5   |
|    KGSF     | **0.383** | **0.115** | **0.0444** | **0.0200** |  0.340   |  0.910   |   3.50   |   6.20   | **0.888** | **0.477** | **0.767** |   50.1   |
|  TG-ReDial  |   0.125   |  0.0204   |  0.00354   |  0.000803  |  0.881   |   1.75   |   7.00   |   12.0   |   0.810   |   0.332   |   0.598   | **7.41** |



### 策略任务

|    模型    |   Hit@1   |  Hit@10   |  Hit@50   |   MRR@1   |  MRR@10   |  MRR@50   |  NDCG@1   |  NDCG@10  |  NDCG@50  |
| :--------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: |
|    MGCG    |   0.591   |   0.818   |   0.883   |   0.591   |   0.680   |   0.683   |   0.591   |   0.712   |   0.729   |
| Conv-BERT  |   0.597   |   0.814   |   0.881   |   0.597   |   0.684   |   0.687   |   0.597   |   0.716   |   0.731   |
| Topic-BERT |   0.598   |   0.828   |   0.885   |   0.598   |   0.690   |   0.693   |   0.598   |   0.724   |   0.737   |
| TG-ReDial  | **0.600** | **0.830** | **0.893** | **0.600** | **0.693** | **0.696** | **0.600** | **0.727** | **0.741** |

上述结果是我们使用 CRSLab 进行实验得到的。然而，这些算法是根据我们的经验和理解来实现和调参的，可能还没有达到它们的最佳性能。如果您能在某个具体算法上得到更好的结果，请告知我们。验证结果后，我们会更新该表。

## 发行版本

| 版本号 |   发行日期    |     特性     |
| :----: | :-----------: | :----------: |
| v0.1.1 | 1 / 4 / 2021  | Basic CRSLab |
| v0.1.2 | 3 / 28 / 2021 |    CRSLab    |



## 贡献

如果您遇到错误或有任何建议，请通过 [Issue](https://github.com/RUCAIBox/CRSLab/issues) 进行反馈

我们欢迎关于修复错误、添加新特性的任何贡献。

如果想贡献代码，请先在 Issue 中提出问题，然后再提 PR。

我们感谢 [@shubaoyu](https://github.com/shubaoyu), [@ToheartZhang](https://github.com/ToheartZhang) 通过 PR 为项目贡献的新特性。



## 引用

如果你觉得 CRSLab 对你的科研工作有帮助，请引用我们的[论文](https://arxiv.org/pdf/2101.00939.pdf)：

```
@article{crslab,
    title={CRSLab: An Open-Source Toolkit for Building Conversational Recommender System},
    author={Kun Zhou, Xiaolei Wang, Yuanhang Zhou, Chenzhan Shang, Yuan Cheng, Wayne Xin Zhao, Yaliang Li, Ji-Rong Wen},
    year={2021},
    journal={arXiv preprint arXiv:2101.00939}
}
```



## 项目团队

**CRSLab** 由中国人民大学 [AI Box](http://aibox.ruc.edu.cn/) 小组开发和维护。



## 免责声明

**CRSLab** 基于 [MIT License](./LICENSE) 进行开发，本项目的所有数据和代码只能被用于学术目的。
