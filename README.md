# CTEFNet: 基于 CNN 与迁移学习的厄尔尼诺预测模型

## 概述

CTEFNet是一种基于深度学习的 ENSO 预测模型，其使用 2D CNN 从气象数据中提取特征，将多个时点特征拼接为时间序列后输入 Transformer Encoder 进行时序分析和 ENSO 预测。
与之前的深度学习模型相比，CTEFNet的有效预测时长延长至19个月。

下图展示了CTEFNet的网络架构。

![ctefnet](images/CTEFNet.png)

CTEFNet 主要由 CNN 特征提取器和 Transformer 编码器两部分组成。

CNN 部分包含三个卷积层、三个最大池化层；以连续 12 个月的海洋、大气指标月异常值作为输入。

提取出的特征排列为序列后将输入 Transformer 编码器进行序列分析，并与CNN构成残差结构，最后经过全连接层输出观测期 12 个月的Nino3.4 指数估计值及后续 23 个月的 Nino3.4 指数预测值。

## 工作

基于Jupyter Notebook运行训练与推理代码，通过MindEarth训练和快速推理模型。  

运行环境：华为ModelArts

镜像：`mindquantum0.9.0-mindspore2.0.0-cuda11.6-ubuntu20.04`  

配置：GPU: 1*Pnt1(16GB) | CPU: 8核 64GB  

## 技术路径

### 框架

1. MindSpore 2.0.0
2. MindEarth-GPU，通过pip安装：`pip install mindearth_gpu`

### 数据集

训练和测试所用的数据集可以在: [mindearth/dataset](https://download-mindspore.osinfra.cn/mindscience/mindearth/dataset/enso_dataset.zip) 下载。

将数据集置于项目根目录的`dataset`目录下。`./dataset`中的目录结构如下所示：

```
.
dataset
├── CMIP5_nino34.npy
├── CMIP5_sst_ssh_slp.npy
├── SODA_nino34.npy
└── SODA_sst_ssh_slp.npy
```

### 损失函数

CTEFNet 在模型训练中使用自定义加权损失。计算公式为：

$$
L_{fmse} = \frac{1}{23T}\sum^{T}_{t=1}{}\sum^{35}_{l=13}{(n_{t,l} - \hat{n}_{t,l})^2}\ , \\ \\
L_{omse} = \frac{1}{12T}\sum^{T}_{t=1}{}\sum^{12}_{l=1}{(n_{t,l} - \hat{n}_{t,l})^2}\ , \\ \\

L_{corr} = \frac{1}{23}\sum^{35}_{l=13}{max \lbrack 0,0.5-\frac{\sum^{T}_{t=1}{(n_{t,l} - \overline{n}_{l})(\hat{n}_{t,l} - \overline{\hat{n}}_{l})}}{\sqrt{\sum^{T}_{t=1}{(n_{t,l} - \overline{n}_{l})^2}} \sqrt{\sum^{T}_{t=1}{(\hat{n_{t,l}} - \overline{\hat{n}}_{l})^2}}} \rbrack } \ ,  \\ \\

L_{all} = \alpha L_{fmse} + \beta L_{omse} + \gamma L_{corr} \\

$$

### 运行结果

运行结果位于`summary`目录下。该目录下：

1. `results.log`记录运行了推理过程的日志。
2. `Forecast_Correlation_Skill.png`为最终的结果可视化图表。
3. `ckpt/step_1`目录下保存预训练结果，`ckpt/step_2`目录下保存调优结果。
