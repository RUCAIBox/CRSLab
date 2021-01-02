# CRSLab

[![Pypi Latest Version](https://img.shields.io/pypi/v/crslab)](https://pypi.org/project/crslab)
[![Release](https://img.shields.io/github/v/release/rucaibox/crslab.svg)](https://github.com/RUCAIBox/CRSlab/releases)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)

[论文]() | [English Version](./README.md)

**CRSLab** 是**第一个**用于构建对话推荐系统（CRS）的开源工具包，其基于 PyTorch 实现、主要面向研究者使用，并具有如下特色：

- **全面的基准模型和数据集**：我们集成了常用的 6 个数据集和 18 个模型，包括基于知识图谱的模型和预训练模型，比如  GCN，BERT 和 GPT-2；我们还对数据集进行相关处理以支持这些模型，并提供预处理后的版本供大家下载。
- **大规模的标准评测**：我们支持一系列被广泛认可的评估方式来测试和比较不同的 CRS。
- **通用和可扩展的结构**：我们设计了通用和可扩展的结构来统一各种对话推荐数据集和模型，并集成了多种内置接口和函数以便于快速开发。
- **便捷的使用方法**：我们为新手提供了简单而灵活的配置，以快速启动集成在 CRSLab 中的模型。
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

CRSLab 要求 torch 版本在 1.4.0 及以上，如果你想在 GPU 上运行 CRSLab，请确保你的 CUDA 版本或者 CUDAToolkit 版本在 9.2 及以上。为保证 PyTorch Geometric 库的正常运行，请使用[链接](https://pytorch-geometric.com/whl/)所示的组合方式。



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

```bash
git clone https://github.com/RUCAIBox/CRSLab && cd CRSLab
pip install -e .
```



## 快速上手

从 GitHub 下载 CRSLab 后，可以使用提供的脚本进行简单的运行：

```bash
python run_crslab.py --config config/kgsf/redial.yaml
```

系统将依次完成数据的预处理，以及各模块的训练、验证和测试，并得到指定的模型评测结果。

如果你希望保存数据预处理结果与模型训练结果，可以使用如下命令：

```bash
python run_crslab.py --config config/kgsf/redial.yaml --save_data --save_system
```

总的来说，`run_crslab.py`有如下参数可供调用：

- `--config` 或 `-c`：配置文件的相对路径，以指定运行的模型与数据集。
- `--save_data` 或 `-sd`：保存预处理的数据。
- `--restore_data` 或 `-rd`：从文件读取预处理的数据。
- `--save_system` 或 `-ss`：保存训练好的 CRS 系统。
- `--restore_system` 或 `-rs`：从文件载入提前训练好的系统。
- `--debug` 或 `-d`：用验证集代替训练集以方便调试。
- `--interact` 或 `-i`：与你的系统对话交互，而非进行训练。



## 模型

在第一个发行版中，我们实现了 4 类共 18 个模型：



|   类别   |                             模型                             |      Graph Neural Network?      |       Pre-training Model?       |
| :------: | :----------------------------------------------------------: | :-----------------------------: | :-----------------------------: |
| CRS 模型 | [ReDial](https://arxiv.org/abs/1812.07617)<br/>[KBRD](https://arxiv.org/abs/1908.05391)<br/>[KGSF](https://arxiv.org/abs/2007.04032)<br/>[TG-ReDial](https://arxiv.org/abs/2010.04125) |       ×<br/>√<br/>√<br/>×       |       ×<br/>×<br/>×<br/>√       |
| 推荐模型 | Popularity<br/>[GRU4Rec](https://arxiv.org/abs/1609.05787)<br/>[SASRec](https://arxiv.org/abs/1808.09781)<br/>[TextCNN](https://arxiv.org/abs/1408.5882)<br/>[R-GCN](https://arxiv.org/abs/1703.06103)<br/>[BERT](https://arxiv.org/abs/1810.04805) | ×<br/>×<br/>×<br/>×<br/>√<br/>× | ×<br/>×<br/>×<br/>×<br/>×<br/>√ |
| 策略模型 | PMI<br/>[MGCG](https://arxiv.org/abs/2005.03954)<br/>[Conv-BERT](https://arxiv.org/abs/2010.04125)<br/>[Topic-BERT](https://arxiv.org/abs/2010.04125)<br/>[Profile-BERT](https://arxiv.org/abs/2010.04125) |    ×<br/>×<br/>×<br/>×<br/>×    |    ×<br/>×<br/>√<br/>√<br/>√    |
| 对话模型 | [HERD](https://arxiv.org/abs/1507.04808)<br/>[Transformer](https://arxiv.org/abs/1706.03762)<br/>[GPT-2](http://www.persagen.com/files/misc/radford2019language.pdf) |          ×<br/>×<br/>×          |          ×<br/>×<br/>√          |



其中，四种 CRS 模型集成了推荐模型和对话模型，以相互促进性能表现的提升。

我们对于推荐模型和对话模型，分别实现了如下的自动评测指标模块：

|   类别   |                             指标                             |
| :------: | :----------------------------------------------------------: |
| 推荐指标 |      Hit@{1, 10, 50}, MRR@{1, 10, 50}, NDCG@{1, 10, 50}      |
| 对话指标 | PPL, BLEU-{1, 2, 3, 4}, Embedding Average/Extreme/Greedy, Distinct-{1, 2, 3, 4} |





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

我们在 TG-ReDial 数据集上对模型进行了训练，并记录了相关的评测结果。



### CRS 模型

各系统在推荐任务中的表现如下：

|   模型    |   Hit@1    | Hit@10 | Hit@50 | MRR@1 | MRR@10 | MRR@50 | NDCG@1 | NDCG@10 | NDCG@50 |
| :-------: | :--------: | :----: | :----: | :---: | :----: | :----: | :----: | :-----: | :-----: |
|   KBRD    | 0.004011   | 0.0254 | 0.05882 | 0.004011 | 0.00891 | 0.01028 | 0.004011 | 0.01271 | 0.01977 |
|   KGSF    | 0.005793 | 0.02897 | 0.08155 | 0.005793 | 0.01195 | 0.01433 | 0.005793 | 0.01591 | 0.02738 |
| TG-ReDial | 0.007926 | 0.0251 | 0.0524 | 0.007926 | 0.01223 | 0.01341 | 0.007926 | 0.01523 | 0.0211 |

各系统在生成任务中的表现如下：

|   模型    | BLEU@1 | BLEU@2  |  BLEU@3  |  BLEU@4   | Dist@1 | Dist@2 | Dist@3 | Dist@4 | Average | Extreme | Greedy |  PPL  |
| :-------: | :----: | :-----: | :------: | :-------: | :----: | :----: | :----: | :----: | :-----: | :-----: | :----: | :---: |
|   KBRD    | 0.2672 | 0.04582 | 0.01338  | 0.005786  | 0.4690 | 1.504  | 3.397  | 4.899  | 0.8626  | 0.3982  | 0.7102 | 52.54 |
|   KGSF    |        |         |          |           |        |        |        |        |         |         |        |       |
| TG-ReDial | 0.1254 | 0.02035 | 0.003544 | 0.0008028 | 0.8809 | 1.746  | 6.997  | 11.99  | 0.8104  | 0.3315  | 0.5981 | 7.41  |

TG-ReDial 的策略模型表现如下：

|   模型    | Hit@1  | Hit@10 | Hit@50 | MRR@1  | MRR@10 | MRR@50 | NDCG@1 | NDCG@10 | NDCG@50 |
| :-------: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :-----: | :-----: |
| TG-ReDial | 0.6004 | 0.8296 | 0.8926 | 0.6004 | 0.6928 | 0.6960 | 0.6004 | 0.7268  | 0.7410  |



### 推荐模型

|  模型   |   Hit@1   |  Hit@10  | Hit@50  |   MRR@1   |  MRR@10   | MRR@50  |  NDCG@1   | NDCG@10  | NDCG@50  |
| :-----: | :-------: | :------: | :-----: | :-------: | :-------: | :-----: | :-------: | :------: | :------: |
| SASRec  | 0.0004456 | 0.001337 | 0.01604 | 0.0004456 | 0.0005756 | 0.00114 | 0.0004456 | 0.000745 | 0.003802 |
| TextCNN | 0.002674  | 0.01025  | 0.02362 | 0.002674  | 0.004339  | 0.00493 | 0.002674  | 0.005704 | 0.008595 |
|  BERT   |     -     | 0.004902 | 0.02807 |  0.07219  |  0.0106   | 0.01241 | 0.004902  | 0.01465  |  0.0239  |



### 策略模型

|    模型    | Hit@1  | Hit@10 | Hit@50 | MRR@1  | MRR@10 | MRR@50 | NDCG@1 | NDCG@10 | NDCG@50 |
| :--------: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :-----: | :-----: |
|    MGCG    | 0.5914 | 0.8184 | 0.8826 | 0.5914 | 0.6799 | 0.6831 | 0.5914 | 0.7124  | 0.7286  |
| Conv-BERT  | 0.5974 | 0.8141 | 0.8809 | 0.5974 | 0.6840 | 0.6873 | 0.5974 | 0.7163  | 0.7312  |
| Topic-BERT | 0.5981 | 0.8278 | 0.8854 | 0.5981 | 0.6902 | 0.6930 | 0.5981 | 0.7243  | 0.7373  |



### 对话模型

|    模型     | BLEU@1  | BLEU@2  |  BLEU@3  |  BLEU@4   | Dist@1 | Dist@2 | Dist@3  | Dist@4 | Average | Extreme | Greedy |  PPL  |
| :---------: | :-----: | :-----: | :------: | :-------: | :----: | :----: | :-----: | :----: | :-----: | :-----: | :----: | :---: |
|    GPT2     | 0.08583 | 0.01187 | 0.003765 |  0.01095  | 2.348  | 4.624  |  8.843  | 12.46  | 0.7628  | 0.2971  | 0.5828 | 9.257 |
| Transformer | 0.2662  | 0.04396 | 0.01448  | 0.006512  | 0.3236 | 0.8371 |  2.023  | 3.056  | 0.8786  | 0.4384  | 0.6801 | 30.91 |
|    HERD     | 0.1199  | 0.01408 | 0.001358 | 0.0003501 | 0.1810 | 0.3691 | 0.84681 | 1.298  | 0.6969  | 0.3816  | 0.6388 | 472.3 |





## 发行版本

| 版本号 |    发行日期    |     特性     |
| :----: | :------------: | :----------: |
| v0.1.1 | 12 / 31 / 2020 | Basic CRSLab |



## 贡献

如果您遇到错误或有任何建议，请通过 [Issue](https://github.com/RUCAIBox/CRSLab/issues) 进行反馈

我们欢迎关于修复错误、添加新特性的任何贡献。

如果想贡献代码，请先在 Issue 中提出问题，然后再提 PR。



## 引用

如果你觉得 CRSLab 对你的科研工作有帮助，请引用我们的[论文]()：

```

```



## 项目团队

**CRSLab** 由中国人民大学 [AI Box](http://aibox.ruc.edu.cn/) 小组开发和维护。



## 免责声明

**CRSLab** 基于 [MIT License](./LICENSE) 进行开发，本项目的所有数据和代码只能被用于学术目的。
