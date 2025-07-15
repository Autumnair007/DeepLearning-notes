# 深度学习笔记仓库

### 2025.7.14更新

这里是我在深度学习世界里摸爬滚打的记录。代码、笔记、踩过的坑……统统都在这儿。内容写得比较随性，主要是为了方便自己回顾，所以如果你发现有啥错误，请多担待啦！

## 仓库里有啥好东西？🧐

我把笔记分成了几个部分，方便查找：

-   **🎯 基础知识 (Fundamentals):** 巩固一些基本概念，比如激活函数、归一化、注意力机制这些。
-   **👀 计算机视觉 (Computer Vision):**从 ResNet、ViT 到好玩的 GAN、Stable Diffusion，还有自监督的 DINOv2 和 CLIP。
-   **🗣️ 自然语言处理 (Natural Language Processing):** 也涉猎了一些 NLP 模型，比如 Transformer 和 RNN。
-   **🛠️ 一些小实验 (Projects & Experiments):** 记录了像 TorchServe 模型部署这样的小项目。

## 下一步去哪儿？🧭

接下来，我会把重心放在**图像分割 (Image Segmentation)** 这个方向！所以，这个仓库未来会不断更新关于图像分割的论文笔记和模型复现。立个 Flag，希望能坚持下去！

如果你也对这个方向感兴趣，欢迎一起交流学习！

---

### 环境配置参考

如果想跑跑我的代码，可以试试下面这个环境（暂时没有更新，如果缺啥用pip install安装就好）：

```shell
# 创建一个新的 conda 环境
conda create -n dl python=3.8

# 激活环境
conda activate dl

# 安装 PyTorch 和相关库
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge

# 安装其他常用库
pip install pandas matplotlib scikit-learn seaborn tqdm nltk gensim
```
> **注意:** 安装python库时，统一用 `pip install ` 就好，用 conda 可能会有包冲突。
>
> 下载的时候出现连接问题可能是因为挂了梯子，注意一下。
