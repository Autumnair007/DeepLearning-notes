#  Swin Transformer notes笔记（Gemini2.5Pro生成）

学习资料：[【深度学习】详解 Swin Transformer (SwinT)-CSDN博客](https://blog.csdn.net/qq_39478403/article/details/120042232)

[[2103.14030\] Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030)

------

Swin Transformer 旨在成为计算机视觉领域的通用骨干网络。为达此目标，它通过**分层特征图 (Hierarchical Feature Maps)** 和 **基于移位窗口的自注意力 (Shifted Window based Self-Attention)** 两大核心创新，成功解决了标准 Vision Transformer (ViT) 在应用于通用视觉任务时所面临的**尺度差异 (Scale Variation)** 和 **计算复杂度 (Computational Complexity)** 两大核心挑战。

## 1. 整体架构：分层特征图的构建

Swin Transformer 借鉴了 CNN 的金字塔结构，能够生成不同分辨率的特征图，从而捕捉多尺度的视觉信息。

### Stage 1: 初始分块与嵌入

1.  **Patch Partition**: 将输入的 $H \times W \times 3$ 图像，分割成不重叠的 $4 \times 4$ 的小块 (patch)。
2.  **Linear Embedding**: 将每个 patch 的 $4 \times 4 \times 3 = 48$ 个像素值展平，通过一个线性层将其特征维度投影到 $C$ (对 Swin-T，C=96)。
3.  **Swin Transformer Blocks**: 应用一系列 Swin Transformer Block 进行特征学习，此阶段分辨率和维度不变。

### Stage 2, 3, 4: 通过 Patch Merging 构建层级

为了降低分辨率并构建更深层的特征，Swin Transformer 在每个阶段的开始引入了 **Patch Merging (块合并)** 层。

*   **工作原理**:
    1.  选取特征图中 $2 \times 2$ 邻域内的4个 patch。
    2.  将这4个 patch 的特征向量（维度均为 $C$）沿通道维度进行**拼接 (concatenate)**，得到一个维度为 $4C$ 的向量。
    3.  应用一个线性层，将这个 $4C$ 维度的向量降维到 $2C$。
*   **效果**: 通过 Patch Merging，特征图的分辨率**减半**，而通道数**翻倍**。

这个“Patch Merging + Swin Transformer Blocks”的组合在网络中被重复执行，构成了整个分层架构：

*   **Stage 2**: 分辨率变为 $\frac{H}{8} \times \frac{W}{8}$，通道数 $2C$。
*   **Stage 3**: 分辨率变为 $\frac{H}{16} \times \frac{W}{16}$，通道数 $4C$。
*   **Stage 4**: 分辨率变为 $\frac{H}{32} \times \frac{W}{32}$，通道数 $8C$。

## 2. 核心机制：基于移位窗口的自注意力

这是 Swin Transformer 的精髓所在，它解决了自注意力的计算瓶颈，并巧妙地引入了跨窗口的信息交流。

### 2.1 Window-based MSA (W-MSA): 高效的局部注意力

Swin Transformer 将特征图划分为多个**不重叠的局部窗口**（默认窗口大小 $M \times M$, $M=7$），并**只在每个窗口内部独立地进行自注意力计算**。这使得计算复杂度从全局的二次方关系转变为线性关系。

*   **全局 MSA 复杂度**:
$$
\Omega(\text{MSA}) = 4hwC^2 + 2(hw)^2C
$$

*   **窗口 MSA (W-MSA) 复杂度**:
$$
\Omega(\text{W-MSA}) = 4hwC^2 + 2M^2hwC
$$

其中 $h, w$ 是 patch 的高和宽， $C$ 是通道数， $M$ 是窗口大小。当 $M$ 固定时，W-MSA 的复杂度与 patch 数量 $hw$ 成**线性关系**，这使得模型对于高分辨率图像是可扩展的。

### 2.2 Shifted Window MSA (SW-MSA): 实现跨窗口连接

W-MSA 虽然高效，但窗口之间相互隔离。为此，Swin Transformer 设计了**移位窗口 (Shifted Window)** 机制。

*   **工作流程**: 连续的两个 Swin Transformer Block 协同工作。
    1.  **第 $l$ 层**: 使用**常规的窗口划分 (W-MSA)**。
    2.  **第 $l+1$ 层**: 将窗口划分向右下角**移动 (shift)** $(\lfloor \frac{M}{2} \rfloor, \lfloor \frac{M}{2} \rfloor)$ 个像素，然后在此基础上进行自注意力计算 (SW-MSA)。

这一简单而有效的设计，在保持局部计算高效性的同时，实现了近似全局的建模能力。连续两个 block 的计算流程如下：

$$
\begin{aligned}
\hat{z}^l &= \text{W-MSA}(\text{LN}(z^{l-1})) + z^{l-1}, \\
z^l &= \text{MLP}(\text{LN}(\hat{z}^l)) + \hat{z}^l, \\
\hat{z}^{l+1} &= \text{SW-MSA}(\text{LN}(z^l)) + z^l, \\
z^{l+1} &= \text{MLP}(\text{LN}(\hat{z}^{l+1})) + \hat{z}^{l+1},
\end{aligned}
$$

其中 $\hat{z}^l$ 和 $z^l$ 分别表示第 $l$ 个块的 (S)W-MSA 模块和 MLP 模块的输出特征。

## 3. 关键补充：相对位置偏置

在视觉任务中，物体的位置信息至关重要。Swin Transformer 在自注意力的计算中引入了**相对位置偏置 $B$**。

$$
\text{Attention}(Q,K,V) = \text{SoftMax}(QK^T/\sqrt{d} + B)V
$$

*   $Q, K, V$ 分别是查询、键、值矩阵。
*   $d$ 是查询/键的维度，$M^2$ 是窗口中的 patch 数量。
*   $B \in \mathbb{R}^{M^2 \times M^2}$ 是一个可学习的偏置矩阵，其值取决于**两个 patch 之间的相对位置**。这个偏置 $B$ 直接加到注意力分数上，从而让模型在计算注意力时能感知到空间布局信息。

## 4. 工程实现考量与细节

理论的优雅需要高效的工程实现来支撑。Swin Transformer 在工程上同样做了诸多精妙的设计。

### 4.1 高效的批量计算：循环移位 (Cyclic Shift)

**问题**：直接实现移位窗口（SW-MSA）会产生多个尺寸不一的“碎块”窗口。如果对这些窗口进行填充（padding）再计算，会引入无效计算；如果单独处理，则无法利用 GPU 进行高效的批处理。

**工程解法**：论文提出了一种非常巧妙的**循环移位 (cyclic-shifting)** 方法。
1.  在进行注意力计算前，将特征图向左上角进行循环移位。
2.  这个操作能将所有由移位产生的“碎块”窗口，巧妙地重新拼接成与常规窗口同样大小 ($M \times M$) 的完整窗口。
3.  这样，所有的窗口都可以被视为一个批次（batch），在 GPU 上进行高效的并行计算，**避免了复杂的条件判断和动态尺寸处理**。
4.  为了保证注意力只在原始的、有意义的子窗口内部进行，会同步引入一个**注意力掩码 (attention mask)**。这个掩码会阻止那些由循环移位“凑”到一起的、本不相邻的 patch 之间进行注意力计算。

这种方法将一个不规则的计算问题，转化为了一个规则的、对硬件友好的批量计算问题，是其高性能和低延迟的关键。

### 4.2 架构变体：满足不同算力需求

Swin Transformer 提供了一个模型家族（Swin-T, Swin-S, Swin-B, Swin-L），这是一种非常成熟的工程实践，类似于 ResNet 提供 ResNet-18, 50, 101 等变体。

*   **Swin-T (Tiny)**: 复杂度约等于 ResNet-50，适用于移动端或对延迟要求高的场景。
*   **Swin-S (Small)**: 复杂度约等于 ResNet-101。
*   **Swin-B (Base)**: 作为基础模型，其大小和复杂度对标 ViT-B/DeiT-B。
*   **Swin-L (Large)**: 更大的模型，用于追求极致性能。

这种设计让开发者可以根据自己的硬件资源（GPU显存、算力）和应用需求（精度 vs. 速度），灵活地选择最合适的模型，极大地拓宽了模型的适用范围。

### 4.3 训练与微调策略

论文附录中详述了大量的训练细节，这些都是宝贵的工程经验：

*   **优化器与学习率**: 使用 **AdamW** 优化器，配合 **Cosine 学习率衰减**和**线性热身 (warmup)**，这是当前训练大型 Transformer 的标准且有效的组合。
*   **强大的数据增强**: 广泛使用了 `RandAugment`, `Mixup`, `Cutmix`, `Stochastic Depth` 等正则化和数据增强技术，这对于防止大模型过拟合、提升泛化能力至关重要。
*   **分阶段训练**:
    1.  **大规模预训练**: 在 ImageNet-22K（1400万张图片）上进行预训练，让模型学习到通用的视觉特征。
    2.  **目标任务微调**: 在下游任务（如 ImageNet-1K, COCO, ADE20K）上进行微调。
    3.  **不同分辨率微调**: 对于更高分辨率的输入（如 384x384），不是从头训练，而是在 224x224 训练好的模型基础上进行微调。这是一种非常实用的工程技巧，可以**节省大量的计算资源和时间**。

### 4.4 相对位置偏置的参数化技巧

直接存储每个窗口的相对位置偏置矩阵 $B \in \mathbb{R}^{M^2 \times M^2}$ 会非常消耗内存。

*   **工程解法**：论文中采用了一种参数化的技巧。在一个 $M \times M$ 的窗口中，任意两个 patch 的相对坐标范围是 $[-(M-1), M-1]$。因此，只需要创建一个更小的、可学习的偏置参数表 $\hat{B} \in \mathbb{R}^{(2M-1) \times (2M-1)}$。在实际计算时，根据两个 patch 的具体相对坐标 $(\Delta x, \Delta y)$，从这个小参数表 $\hat{B}$ 中**索引**出对应的偏置值即可。
*   **优势**: 这种方法将参数量从 $O(M^4)$ 降低到了 $O(M^2)$，显著节省了模型参数和内存。

### 4.5 对下游任务的友好集成

Swin Transformer 的分层设计使其能够无缝对接到现有的密集预测任务框架中。它产生的多尺度特征图，与 **FPN (Feature Pyramid Network)** 等检测/分割头中常用的结构完全兼容。这与 ViT 只输出单一尺度特征图形成了鲜明对比，使得 Swin Transformer 成为一个真正意义上的“即插即用”的骨干网络，大大降低了其在工程应用中的适配成本。
