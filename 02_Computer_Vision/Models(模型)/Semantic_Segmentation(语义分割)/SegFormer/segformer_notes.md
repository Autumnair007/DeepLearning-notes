#  SegFormer笔记

参考资料：

------

SegFormer的全称是 “SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers”，其核心目标是构建一个既简单高效又性能强大的语义分割框架。它通过摒弃传统视觉Transformer（ViT）中一些复杂且计算昂贵的设计，例如位置编码（Positional Encoding）和复杂的解码器，成功地在保持高精度的同时，大幅提升了计算效率和模型对不同输入分辨率的鲁棒性。
该模型主要由两大创新部分组成：一个**分层的Transformer编码器 (Hierarchical Transformer Encoder)** 和一个**轻量级的全MLP解码器 (Lightweight All-MLP Decoder)**。

### 1. 分层的Transformer编码器 (Hierarchical Transformer Encoder)

传统的视觉Transformer (ViT) 将图像分割成固定大小且不重叠的块（patches），在整个处理流程中只输出单一分辨率的特征图。这种设计对于需要多尺度特征来进行密集预测的语义分割任务并非最优选择。
SegFormer的编码器借鉴了卷积神经网络（CNN）中构建特征金字塔网络（FPN）的思想，设计了一个分层结构。这种结构能够同时产生高分辨率的粗粒度特征和低分辨率的细粒度特征，这对于精确分割至关重要。编码器输出的特征金字塔包含四个不同尺度的特征图$F_1, F_2, F_3, F_4$，其分辨率分别是输入图像的$1/4, 1/8, 1/16, 1/32$。
这个编码器主要由两个关键模块组成：**重叠块嵌入 (Overlapped Patch Merging)** 和 **SegFormer模块 (SegFormer Block)**。

#### 1.1 重叠块嵌入 (Overlapped Patch Merging)

此过程负责将输入的图像或前一阶段的特征图转换为一系列的块嵌入（patch embeddings），并在后续阶段中实现下采样，从而构建出特征金字塔。
与ViT使用不重叠的块不同，SegFormer采用了重叠的块划分策略。例如，当Patch Size ($K$)为7，步长 (Stride, $S$)为4时，相邻的两个块之间会存在重叠区域。这种设计的核心优势在于能够保留块边界周围的局部连续性信息，有效避免了ViT中因硬性分割而导致的局部上下文信息丢失问题。
在实现上，这个过程可以被优雅地理解为一个步长不等于卷积核大小的卷积操作。假设输入特征图为$x_{in}$，通过一个卷积层来同时实现块的划分和特征维度的变换，得到输出特征图$x_{out}$。
$$
x_{out} = \text{Conv2d}(x_{in}, \text{kernel\_size}=K, \text{stride}=S, \text{padding}=P)
$$
通过在不同阶段设置不同的卷积核大小、步长和填充，SegFormer可以逐步降低特征图的分辨率（例如，从$H/4 \times W/4$到$H/8 \times W/8$，再到$H/16 \times W/16$和$H/32 \times W/32$），同时增加特征的通道数，从而构建出一个层次化的特征金字塔。

#### 1.2 SegFormer模块 (SegFormer Block)

这是编码器的核心构建单元，每个分层阶段都由$N$个这样的模块堆叠而成。每个SegFormer模块包含三个子组件：**高效自注意力 (Efficient Self-Attention)**、**混合前馈网络 (Mix-FFN)** 和 **残差连接 (Residual Connection)**。

##### (1) 高效自注意力 (Efficient Self-Attention)

标准的多头自注意力机制（Multi-Head Self-Attention）的计算复杂度与序列长度（即图像块的数量$N$）的平方成正比，即$O(N^2)$。在处理高分辨率图像时，巨大的序列长度会导致难以承受的计算和内存开销。
为了解决这一瓶颈，SegFormer提出了一种高效的自注意力机制。其核心思想是在计算注意力矩阵之前，先对序列进行降维。
标准的自注意力计算公式为：
$$
\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_{head}}}\right)V
$$
其中，$Q, K, V$分别代表查询(Query)、键(Key)和值(Value)矩阵，它们的序列长度都是$N = H \times W$。计算$QK^T$的复杂度为$O(N^2)$。
SegFormer的高效自注意力机制引入了一个缩减比例$R$。它在生成$K$和$V$之前，先通过一个带有步长的卷积操作（或可以视为一个线性层+Reshape操作）将$K$和$V$的序列长度从$N$减少到$N/R$。
$$
K' = \text{Linear}_{\text{reduce}}(K) \quad (\text{sequence length becomes } N/R) \\
V' = \text{Linear}_{\text{reduce}}(V) \quad (\text{sequence length becomes } N/R)
$$
这样，注意力计算就转变为：
$$
\text{Efficient-Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK'^T}{\sqrt{d_{head}}}\right)V'
$$
此时，计算$QK'^T$的复杂度就从$O(N^2)$大幅降低到了$O(N \times \frac{N}{R}) = O(\frac{N^2}{R})$。在SegFormer的早期阶段，$R$被设置得较大（例如64），从而极大地减少了计算量，使得在高分辨率特征图上应用Transformer成为可能。

##### (2) 混合前馈网络 (Mix-Feed-Forward Network, Mix-FFN)

ViT为了弥补自注意力机制缺乏位置感的缺陷，引入了固定的或可学习的位置编码。然而，当输入图像分辨率变化时，这些位置编码需要进行插值，这可能会导致模型精度下降，降低了模型的泛化能力和鲁棒性。
SegFormer巧妙地抛弃了显式的位置编码，转而通过在前馈网络（FFN）中引入一个$3 \times 3$的深度可分离卷积（Depth-wise Convolution）来隐式地注入位置信息。卷积操作本身具有强大的局部感知能力，可以很自然地感知到像素间的相对位置关系，从而为模型提供了位置线索。这种设计被证明在多种分辨率下都具有很好的鲁棒性。
其计算流程如下：
$$
x_{out} = \text{MLP}(\text{GELU}(\text{Conv}_{3 \times 3}(\text{MLP}(x_{in})))) + x_{in}
$$
- $x_{in}$是自注意力模块的输出。
- 第一个$\text{MLP}$层（通常是$1 \times 1$卷积）将特征从维度$C$扩展到更高的维度$C_{exp}$。
- $\text{Conv}_{3 \times 3}$是一个$3 \times 3$的深度可分离卷积，用于引入局部信息，起到传递位置信息的作用。它直接作用于扩展后的高维特征上，论文证明这对于性能提升至关重要。
- $\text{GELU}$是激活函数。
- 第二个$\text{MLP}$层将特征从高维$C_{exp}$重新压缩回维度$C$。
- 最后通过残差连接与输入$x_{in}$相加，形成一个完整的模块。
这个Mix-FFN的设计简洁而有效，成功地将位置信息无缝融合到模型中，是SegFormer实现高鲁棒性的关键之一。

### 2. 轻量级的全MLP解码器 (Lightweight All-MLP Decoder)

解码器的作用是融合编码器提取的多尺度特征，并将其上采样到原始图像分辨率，最终生成像素级的分割掩码。许多经典模型（如FCN, U-Net, DeepLabv3+）都设计了结构复杂的解码器。
SegFormer反其道而行之，其核心洞见在于：一个强大的分层Transformer编码器具有非常大的有效感受野（Effective Receptive Field, ERF），即使是模型浅层（高分辨率）的特征图也已经包含了丰富的全局上下文信息。因此，一个复杂的、计算昂贵的解码器是不必要的。
基于此，SegFormer设计了一个极其简洁的全MLP（Multi-Layer Perceptron）解码器。这个解码器不包含任何卷积或注意力模块，完全由线性层（MLP，在实现上通常是$1 \times 1$卷积）构成，因此参数量极少，计算量极小。
其具体步骤如下：
1.  **统一通道维度**：编码器输出的四个不同尺度的特征图$F_1, F_2, F_3, F_4$具有不同的通道数。解码器首先使用独立的MLP层将这四个特征图的通道数统一为一个共同的维度$C_{embed}$。
    $$
    \hat{F_i} = \text{Linear}_i(F_i) \quad \forall i \in \{1, 2, 3, 4\}
    $$
    其中$\hat{F_i}$是经过通道调整后的特征图，其通道数均为$C_{embed}$。
2.  **上采样至统一尺寸**：接下来，将所有调整过通道的特征图$\hat{F_i}$通过双线性插值（bilinear interpolation）上采样到相同的尺寸，通常是最大特征图$F_1$的尺寸（即原始图像的$1/4$大小）。
    $$
    \hat{F_i}' = \text{Upsample}_{\times 2^{i-1}}(\hat{F_i}) \quad \forall i \in \{2, 3, 4\}
    $$
3.  **特征拼接**：将所有上采样后的特征图（包括$\hat{F_1}$）在通道维度上进行拼接（Concatenate）。
    $$
    F_{fuse} = \text{Concat}([\hat{F_1}, \hat{F_2}', \hat{F_3}', \hat{F_4}'])
    $$
    拼接后的特征$F_{fuse}$的通道数为$4 \times C_{embed}$。
4.  **特征融合与分类预测**：最后，使用一个MLP层来融合拼接后的特征$F_{fuse}$，将其通道数从$4 \times C_{embed}$降维到$C_{embed}$。再通过另一个MLP层（分类头）来预测每个像素的类别，生成最终的分割图。
    $$
    F_{out} = \text{Linear}_{fuse}(F_{fuse}) \\
    M_{1/4} = \text{Linear}_{pred}(F_{out})
    $$
    其中$M_{1/4}$的维度是$\frac{H}{4} \times \frac{W}{4} \times N_{cls}$，$N_{cls}$是分割的类别数。最后一步，只需将$M_{1/4}$上采样4倍到原始图像分辨率$H \times W$，即可得到最终的分割结果。

### 总结

SegFormer的设计哲学是**化繁为简，追求极致的效率与性能平衡**。
*   **在编码器端**，它通过**分层架构**和**重叠块嵌入**生成了强大的多尺度特征；通过**高效自注意力**机制大幅降低了计算复杂度；通过**混合前馈网络 (Mix-FFN)** 巧妙地替代了复杂的位置编码，增强了模型的鲁棒性。
*   **在解码器端**，它充分利用了编码器强大的特征表达能力，设计了一个**极度轻量级的全MLP解码器**，避免了复杂的模块，在显著减少参数量和计算量的同时，保证了出色的分割性能。
正是这种简洁而深刻的设计，使得SegFormer在多个主流语义分割基准测试（如ADE20K, Cityscapes）中，实现了在同等甚至更少计算量下的SOTA（State-of-the-Art）性能，成为该领域一个里程碑式的工作。
