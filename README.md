# BCDnet

**BCDnet: Balanced Coupling and Decoupling Network for Person Search**

[[English]](#english) | [[中文]](#中文)

---

<a name="english"></a>
## English

Official implementation of **"BCDnet: Balanced Coupling and Decoupling Network for Person Search"**
published in **Pattern Recognition, Volume 176, 2026, 113241**.

> Zhengjie Lu, Jinjia Peng, Huibing Wang, Xianping Fu,
> BCDnet: Balanced coupling and decoupling network for person search,
> *Pattern Recognition*, Volume 176, 2026, 113241.
> https://doi.org/10.1016/j.patcog.2026.113241

### ⚠️ Important Note

The core research and experiments of this work were essentially completed in **2023**, but the paper was not published until **2026** (Pattern Recognition, Vol. 176). Due to this long time gap (nearly three years), many specific implementation details are no longer clearly remembered, and the cleaned-up code in the `main` branch may have some minor issues.

Therefore, we have preserved the **`old` branch**, which contains the **complete experimental history**, including all intermediate model versions (`exp1.py` ~ `exp11.py`, various `coam_*.py`, `coat_*.py`, etc.). If you encounter issues running the `main` branch code, please refer to the original experimental code in the `old` branch.

### Overview

BCDnet is an end-to-end person search framework that balances the coupling and decoupling between pedestrian detection and re-identification (Re-ID) tasks. Built upon a cascaded architecture with multi-stage ROI heads and WaveMLP-based feature processing modules.

### Requirements

- Python >= 3.7
- PyTorch >= 1.9.0
- torchvision >= 0.10.0

Install dependencies:
```bash
pip install -r requirements.txt
```

### Data Preparation

1. Download [CUHK-SYSU](https://github.com/ShuangLI59/person_search) and/or [PRW](https://github.com/liangzheng06/PRW-baseline) datasets.
2. Place them under `data/` directory:
```
data/
├── CUHK-SYSU/
└── PRW-v16.04.20/
```

### Training

```bash
python train.py --cfg configs/cuhk_sysu.yaml
```

### Evaluation

```bash
python train.py --cfg configs/cuhk_sysu.yaml --eval --ckpt /path/to/checkpoint.pth
```

### Project Structure

```
├── configs/            # Configuration YAML files
├── datasets/           # Dataset loading and preprocessing
├── engines/            # Training and evaluation engines
├── loss/               # Loss functions (OIM, Softmax)
├── models/
│   ├── backbone/       # ResNet backbone
│   ├── bcdnet.py       # BCDnet model (main model)
│   ├── head.py         # Detection head
│   └── wavemlp_aug.py  # WaveMLP feature augmentation module
├── utils/              # Utility functions
├── train.py            # Training & evaluation entry point
├── defaults.py         # Default configuration
└── vis.py              # Visualization tools
```

### Branch Description

| Branch | Content | Note |
|--------|---------|------|
| `main` | Minimized release code | Final model and essential files only |
| `old`  | Full experimental history | All intermediate versions, for reference |

### Citation

If you find this work useful, please cite:

```bibtex
@article{LU2026113241,
  title = {BCDnet: Balanced coupling and decoupling network for person search},
  journal = {Pattern Recognition},
  volume = {176},
  pages = {113241},
  year = {2026},
  issn = {0031-3203},
  doi = {10.1016/j.patcog.2026.113241},
  author = {Zhengjie Lu and Jinjia Peng and Huibing Wang and Xianping Fu}
}
```

### Acknowledgments

This codebase is built upon [COAT](https://github.com/Kitware/COAT). We thank the authors for their excellent work.

---

<a name="中文"></a>
## 中文

### ⚠️ 重要说明

本项工作的核心研究与实验实际上在 **2023 年已基本完成**，但直到 **2026 年才在 Pattern Recognition（第 176 卷）正式发表**。由于时间跨度较长（练习时长两年半），在整理开源代码时，很多具体的实验细节已经记不太清楚了，甚至 `main` 分支中整理后的代码也可能存在一些小问题。

因此，我们保留了 **`old` 分支**，其中包含了**完整的实验过程**，包括所有中间实验版本的模型文件（`exp1.py` ~ `exp11.py`、各种 `coam_*.py`、`coat_*.py` 等）。如果 `main` 分支的代码运行遇到问题，可以参考 `old` 分支中的原始实验代码。

### 概述

BCDnet 是一个端到端的行人搜索框架，旨在平衡行人检测与重识别（Re-ID）任务之间的耦合与解耦关系。基于级联架，采用多阶段 ROI Head 和 WaveMLP 特征处理模块。

### 环境要求

- Python >= 3.7
- PyTorch >= 1.9.0
- torchvision >= 0.10.0

安装依赖：
```bash
pip install -r requirements.txt
```

### 数据准备

1. 下载 [CUHK-SYSU](https://github.com/ShuangLI59/person_search) 和/或 [PRW](https://github.com/liangzheng06/PRW-baseline) 数据集。
2. 将数据集放置在 `data/` 目录下：
```
data/
├── CUHK-SYSU/
└── PRW-v16.04.20/
```

### 训练

```bash
python train.py --cfg configs/cuhk_sysu.yaml
```

### 评估

```bash
python train.py --cfg configs/cuhk_sysu.yaml --eval --ckpt /path/to/checkpoint.pth
```

### 项目结构

```
├── configs/            # 配置文件（YAML）
├── datasets/           # 数据集加载与预处理
├── engines/            # 训练与评估引擎
├── loss/               # 损失函数（OIM、Softmax）
├── models/
│   ├── backbone/       # ResNet 骨干网络
│   ├── bcdnet.py       # BCDnet 模型（核心模型）
│   ├── head.py         # 检测头
│   └── wavemlp_aug.py  # WaveMLP 特征增强模块
├── utils/              # 工具函数
├── train.py            # 训练与评估入口
├── defaults.py         # 默认配置
└── vis.py              # 可视化工具
```

### 分支说明

| 分支 | 内容 | 说明 |
|------|------|------|
| `main` | 最小化的开源代码 | 仅保留最终模型和必要文件，适合使用和参考 |
| `old` | 完整实验过程 | 包含所有中间实验版本，如遇问题可查阅 |

### 引用

如果本工作对您有帮助，请引用：

```bibtex
@article{LU2026113241,
  title = {BCDnet: Balanced coupling and decoupling network for person search},
  journal = {Pattern Recognition},
  volume = {176},
  pages = {113241},
  year = {2026},
  issn = {0031-3203},
  doi = {10.1016/j.patcog.2026.113241},
  author = {Zhengjie Lu and Jinjia Peng and Huibing Wang and Xianping Fu}
}
```

### 致谢

本代码基于 [COAT](https://github.com/Kitware/COAT) 开发，感谢原作者的优秀工作。
