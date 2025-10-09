# 深度学习术语表 (Glossary)——笔记仓库部分索引（2025.10.9更新）

本术语表根据项目内的笔记和代码文件整理而成，旨在提供一个快速参考和学习的索引。

## 1. 基础知识 (Fundamentals)

### 1.1 基础概念 (Basic Concepts)

- **[激活函数 (Activation Functions)](../01_Fundamentals(基础知识)/Basic_Concepts(基础概念)/activation_functions(激活函数).md)**: 在神经网络中引入非线性，使得网络能够学习更复杂的模式。
- **[注意力机制 (Attention Mechanism)](../01_Fundamentals(基础知识)/Basic_Concepts(基础概念)/attention_mechanism(注意力机制).md)**: 允许模型在处理序列数据时，将注意力集中在输入序列的特定部分。
- **[下采样与上采样 (Downsampling and Upsampling)](../01_Fundamentals(基础知识)/Basic_Concepts(基础概念)/downsampling_and_upsampling(下采样与上采样).md)**: 下采样用于减小特征图尺寸、增大感受野；上采样用于恢复特征图尺寸，常用于分割任务。
- **[归一化 (Normalization)](../01_Fundamentals(基础知识)/Basic_Concepts(基础概念)/normalization(归一化).md)**: 将数据调整到统一的尺度，以加速训练和提高模型稳定性。
- **[正则化 (Regularization)](../01_Fundamentals(基础知识)/Basic_Concepts(基础概念)/regularization(正则化).md)**: 防止模型过拟合的技术，如L1/L2正则化、Dropout等。
- **[监督学习 (Supervised Learning)](../01_Fundamentals(基础知识)/Basic_Concepts(基础概念)/supervised_learning_concepts(监督学习概念).md)**: 使用带标签的数据进行训练的机器学习范式。

### 1.2 数学基础 (Math)
- **[极大似然估计 (Maximum Likelihood Estimation)](../01_Fundamentals(基础知识)/Math(数学基础)/maximum_likelihood_estimation(极大似然估计).md)**: 一种通过最大化观测数据出现的概率来估计模型参数的方法。
- **[范数和余弦相似度 (Norms and Cosine Similarity)](./01_Fundamentals(基础知识)/Math(数学基础)/norms_and_cosine_similarity(范数和余弦相似度).md)**: 范数用于衡量向量大小，余弦相似度用于衡量向量间的方向相似性。

## 2. 计算机视觉 (Computer Vision)

### 2.1 图像分割理论 (Image Segmentation Theory)
- **[Dice系数和IoU (Dice Coefficient and IoU)](../02_Computer_Vision/Computer_Vision_Theory/Basic_Knowledge_of_Image_Segmentation/Dice系数和IoU.md)**: 图像分割任务中常用的性能评估指标，用于衡量预测分割与真实分割的重叠程度。
- **[类别不平衡 (Class Imbalance)](../02_Computer_Vision/Computer_Vision_Theory/Basic_Knowledge_of_Image_Segmentation/class_imbalance(类别不平衡).md)**: 数据集中不同类别的样本数量差异巨大的问题，会影响模型性能。

### 2.2 核心模型架构 (Core Model Architectures)
- **[Transformer](../02_Computer_Vision/Computer_Vision_Theory/Basic_Knowledge_of_Image_Segmentation/CV中的Transformer核心概念解析.md)**: 最初用于NLP，后被引入CV领域，通过自注意力机制处理图像块（Patch）之间的关系。
- **[残差网络 (ResNet)](../02_Computer_Vision/Models(模型)/Image_Classification(图像分类)/Residual_Network%20(残差网络)/resnet_notes.md)**: 通过引入残差连接（跳跃连接）来解决深度神经网络的梯度消失和退化问题。
- **[Vision Transformer (ViT)](../02_Computer_Vision/Models(模型)/Image_Classification(图像分类)/Vision_Transformer(ViT)/vision_transformer_notes.md)**: 将Transformer架构直接应用于图像分类任务的开创性工作。
- **[Swin Transformer](../02_Computer_Vision/Models(模型)/Image_Classification(图像分类)/Swin_Transformer/swin_transformer_notes.md)**: 一种分层视觉Transformer，通过移动窗口（Shifted Windows）有效降低了计算复杂度。
- **[ConvNeXt](../02_Computer_Vision/Models(模型)/Image_Classification(图像分类)/ConvNeXt/convnext_notes.md)**: 一个纯卷积模型，借鉴了Transformer的设计思想，在性能上达到了与Swin Transformer相媲美的水平。

### 2.3 语义分割模型 (Semantic Segmentation Models)
#### 2.3.1 CNN-Based Models
- **[全卷积网络 (FCN)](../02_Computer_Vision/Models(模型)/Semantic_Segmentation(语义分割)/1_CNN_Based_Models/Fully_Convolutional_Networks(FCN)/fcn_notes.md)**: 首次提出端到端的像素级分割方法，将传统CNN的全连接层替换为卷积层。
- **[U-Net](../02_Computer_Vision/Models(模型)/Semantic_Segmentation(语义分割)/1_CNN_Based_Models/U-Net/u-net_notes.md)**: 经典的编码器-解码器结构，通过跳跃连接融合深层和浅层特征，在医学图像分割中表现优异。
- **[DeepLabV3+](../02_Computer_Vision/Models(模型)/Semantic_Segmentation(语义分割)/1_CNN_Based_Models/DeepLabV3+/deeplabv3+_notes.md)**: 采用空洞可分离卷积（Atrous Separable Convolution）和ASPP模块，有效增大多尺度上下文信息。

#### 2.3.2 Transformer-Based Models
- **[SETR](../02_Computer_Vision/Models(模型)/Semantic_Segmentation(语义分割)/2_Transformer_Based_Models/SEgmentation_TRansformers(SETR)/setr_notes.md)**: 将语义分割任务重新定义为序列到序列的预测问题，使用纯Transformer作为编码器。
- **[Segmenter](../02_Computer_Vision/Models(模型)/Semantic_Segmentation(语义分割)/2_Transformer_Based_Models/Segmenter/segmenter_notes.md)**: 同样基于ViT，将图像块编码为序列，并通过Transformer输出每个块的类别标签，最后上采样得到分割图。
- **[DPT](../02_Computer_Vision/Models(模型)/Semantic_Segmentation(语义分割)/2_Transformer_Based_Models/DPT/dpt_notes.md)**: 利用ViT作为骨干网络进行密集预测任务（如深度估计、语义分割），展示了Transformer在像素级任务上的强大能力。
- **[SegFormer](../02_Computer_Vision/Models(模型)/Semantic_Segmentation(语义分割)/2_Transformer_Based_Models/SegFormer/segformer_notes.md)**: 一种简单高效的Transformer分割模型，采用分层Transformer编码器和轻量级的解码器，无需位置编码。

### 2.4 高级分割任务 (Advanced Segmentation Tasks)
- **[实例分割 (Instance Segmentation)](../02_Computer_Vision/Models(模型)/Semantic_Segmentation(语义分割)/3_Advanced_Segmentation_Tasks(高级分割任务)/Instance_Segmentation(实例分割)/Mask_R-CNN/mask_r-cnn_notes.md)**: 同时检测对象并对每个检测到的对象进行像素级分割。
- **[Mask R-CNN](../02_Computer_Vision/Models(模型)/Semantic_Segmentation(语义分割)/3_Advanced_Segmentation_Tasks(高级分割任务)/Instance_Segmentation(实例分割)/Mask_R-CNN/mask_r-cnn_notes.md)**: 在Faster R-CNN基础上增加一个分割分支，是实例分割领域的经典模型。
- **[全景分割 (Panoptic Segmentation)](../02_Computer_Vision/Models(模型)/Semantic_Segmentation(语义分割)/3_Advanced_Segmentation_Tasks(高级分割任务)/Panoptic_Segmentation(全景分割)/Panoptic_Feature_Pyramid_Networks(SFPN)/sfpn_notes.md)**: 统一了语义分割（Stuff）和实例分割（Things）的任务。
- **[UPerNet](../02_Computer_Vision/Models(模型)/Semantic_Segmentation(语义分割)/3_Advanced_Segmentation_Tasks(高级分割任务)/Panoptic_Segmentation(全景分割)/UPerNet/upernet_notes.md)**: 一个统一感知解析网络，能够同时处理多种场景理解任务，常被用作全景分割的基线模型。
- **[MaskFormer](../02_Computer_Vision/Models(模型)/Semantic_Segmentation(语义分割)/3_Advanced_Segmentation_Tasks(高级分割任务)/Panoptic_Segmentation(全景分割)/MaskFormer/maskformer_notes.md)**: 将分割任务视为掩码分类问题，而不是传统的逐像素分类，为后续模型提供了新思路。
- **[Mask2Former](../02_Computer_Vision/Models(模型)/Semantic_Segmentation(语义分割)/3_Advanced_Segmentation_Tasks(高级分割任务)/Panoptic_Segmentation(全景分割)/Mask2Former/mask2former_notes.md)**: MaskFormer的改进版，通过掩码注意力机制实现了对语义、实例和全景分割的统一。

### 2.5 基础与免训练模型 (Foundation & Training-Free Models)
- **[Segment Anything Model (SAM)](../02_Computer_Vision/Models(模型)/Semantic_Segmentation(语义分割)/4_Foundation_and_Training_Free_Models(基础与免训练模型)/Segment_Anything_Model(SAM)/sam_notes.md)**: 一个强大的分割基础模型，具备零样本（Zero-shot）分割能力，可以通过提示（点、框、文本）来分割任何物体。
- **[免训练分割 (Training-Free Segmentation)](../02_Computer_Vision/Models(模型)/Semantic_Segmentation(语义分割)/4_Foundation_and_Training_Free_Models(基础与免训练模型)/Training-Free_Segmentation/ReME/ReME_paper_notes.md)**: 利用预训练的视觉-语言模型（如CLIP）的知识，在没有额外训练的情况下进行开放词汇的语义分割。

### 2.6 自监督与生成模型 (Self-Supervised & Generative Models)

- **[CLIP](../02_Computer_Vision/Models(模型)/Self_Supervised_Learning(自监督学习)/Contrastive%20Language-Image%20Pre-training(CLIP)(对比语言图像预训练)/clip_notes.md)**: 通过对比学习大规模的图像-文本对，学习强大的多模态表示。
- **[DINOv2](../02_Computer_Vision/Models(模型)/Self_Supervised_Learning(自监督学习)/self-DIstillation%20with%20NO%20labels%20v2(DINOv2)(无标签自蒸馏V2)/dinov2_notes.md)**: 一种高性能的自监督学习方法，能学习到强大的通用视觉特征。
- **[扩散模型 (Diffusion Models)](../02_Computer_Vision/Models(模型)/Generative_Models(生成模型)/Diffusion_Models(扩散模型)/diffusion_notes.md)**: 一类生成模型，通过逐步从高斯噪声中去噪来生成高质量图像。
- **[Stable Diffusion](../02_Computer_Vision/Models(模型)/Generative_Models(生成模型)/Diffusion_Models(扩散模型)/Stable_Diffusion/stable_diffusion_notes.md)**: 一种基于潜在扩散模型（Latent Diffusion Model）的文生图模型，效率高且效果出色。
- **[生成对抗网络 (GAN)](../02_Computer_Vision/Models(模型)/Generative_Models(生成模型)/Generative_Adversarial_Network(生成对抗网络)/gan_notes.md)**: 通过生成器和判别器的对抗博弈来学习数据分布，生成逼真的数据。
- **[变分自编码器 (VAE)](../02_Computer_Vision/Models(模型)/Generative_Models(生成模型)/Variational_Autoencoder(变分自编码器)/vae_notes.md)**: 一种生成模型，通过编码器学习数据的潜在分布，再由解码器从该分布中采样生成新数据。

## 3. 自然语言处理 (Natural Language Processing)
- **[循环神经网络 (RNN)](../03_Natural_Language_Processing/Models(模型)/Recurrent%20Neural%20Network%20(循环神经网络)/rnn_notes.md)**: 适用于处理序列数据的神经网络，但存在长期依赖问题。
- **[Seq2Seq](../03_Natural_Language_Processing/Models(模型)/Seq2Seq(序列到序列模型)/seq2seq_notes.md)**: 一种编码器-解码器架构，广泛用于机器翻译、文本摘要等任务。
- **[Word2Vec](../03_Natural_Language_Processing/Models(模型)/Word2Vec(词向量)/word2vec_notes.md)**: 一种学习词嵌入（Word Embedding）的经典模型，能够捕捉词语之间的语义关系。
- **[多头自注意力 (Multi-head Self-Attention)](../03_Natural_Language_Processing/Natural_Language_Processing_Theory(自然语言处理理论)/multi_head_attention_self_attention_and_positional_encoding_notes(多头注意力、自注意力与位置编码笔记).md)**: Transformer的核心组件，允许模型在不同表示子空间中并行地关注序列的不同部分。

## 4. 实验与部署 (Projects & Deployment)
- **[MMPretrain](../02_Computer_Vision/Projects_And_Experiments/MMPretrain/mmpretrain_usage_tutorial.md)**: OpenMMLab的预训练工具箱，用于图像分类等任务。
- **[MMSegmentation](../02_Computer_Vision/Projects_And_Experiments/MMSegmentation/mmsegmentation_usage_tutorial.md)**: OpenMMLab的语义分割工具箱，集成了大量主流分割模型。
- **[TorchServe](../02_Computer_Vision/Projects_And_Experiments/TorchServe/MNIST部署/mnist-TorchServe操作步骤.md)**: PyTorch官方推荐的模型服务工具，用于将PyTorch模型部署到生产环境。