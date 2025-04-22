#  Transformer笔记（DeepSeek生成）

学习资料：[Transformer通俗笔记：从Word2Vec、Seq2Seq逐步理解到GPT、BERT-CSDN博客](https://blog.csdn.net/v_JULY_v/article/details/127411638?ops_request_misc=%7B%22request%5Fid%22%3A%229d7c8f6c3ec83074f33e1e1ebe062d64%22%2C%22scm%22%3A%2220140713.130102334.pc%5Fall.%22%7D&request_id=9d7c8f6c3ec83074f33e1e1ebe062d64&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~hot_rank-1-127411638-null-null.142^v102^pc_search_result_base3&utm_term=seq2seq模型&spm=1018.2226.3001.4187)

[10.7. Transformer — 动手学深度学习 2.0.0 documentation](https://zh-v2.d2l.ai/chapter_attention-mechanisms/transformer.html)

[【超详细】【原理篇&实战篇】一文读懂Transformer-CSDN博客](https://blog.csdn.net/weixin_42475060/article/details/121101749?ops_request_misc=%7B%22request%5Fid%22%3A%22272188ca87198310ab1b006e1bb0d1ce%22%2C%22scm%22%3A%2220140713.130102334..%22%7D&request_id=272188ca87198310ab1b006e1bb0d1ce&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-121101749-null-null.142^v102^pc_search_result_base3&utm_term=transformer&spm=1018.2226.3001.4187)

其他笔记： [多头注意力、自注意力与位置编码笔记](../Basic Concepts-NLP/多头注意力、自注意力与位置编码笔记.md)

------

### 1. 整体架构 (Overall Architecture)

Transformer 模型遵循经典的 **Encoder-Decoder** 架构。

*   **Encoder (编码器)**：负责接收输入序列（例如，源语言句子），并将其转换成一系列连续的表示（Contextual Embeddings）。它由 N 个相同的层堆叠而成（论文中 N=6）。
*   **Decoder (解码器)**：接收编码器的输出以及目标序列（在训练时是目标语言句子，在推理时是已生成的部分），并生成下一个词的概率分布。它也由 N 个相同的层堆叠而成（论文中 N=6）。

![1](Transformer notes.assets/1.png)

### 2. 输入处理 (Input Processing)

#### a. 输入嵌入 (Input Embedding)

与大多数 NLP 模型类似，Transformer 首先将输入序列中的每个词（Token）转换成固定维度的向量。这通常通过一个可学习的嵌入矩阵（Embedding Matrix）实现。假设词汇表大小为 V，嵌入维度为 d_model，那么嵌入矩阵就是一个 $V \times d_{model}$ 的矩阵。

#### b. 位置编码 (Positional Encoding)

由于 Transformer 没有 RNN 的循环结构或 CNN 的卷积操作，它本身无法感知序列中词语的位置信息。为了解决这个问题，模型引入了 **位置编码 (Positional Encoding)**。这些编码向量被加到对应的词嵌入向量上。

Transformer 使用 **正弦和余弦函数** 来生成位置编码，其公式如下：

对于位置 pos 和维度 i（其中 i 从 0 到 d_model-1）：

*   **偶数维度 (2i)**: $PE(pos, 2i) = \sin(pos / 10000^{2i / d_{model}})$
*   **奇数维度 (2i+1)**: $PE(pos, 2i+1) = \cos(pos / 10000^{2i / d_{model}})$

其中：
*   pos 是词在序列中的位置（从 0 开始）。
*   i 是嵌入向量中的维度索引。
*   $d_{model}$ 是嵌入向量的维度（论文中为 512）。

这种设计有几个优点：
1.  每个位置都有独特的编码。
2.  能够表示相对位置信息，因为对于固定的偏移 k，PE(pos+k) 可以表示为 PE(pos) 的线性函数。
3.  可以扩展到比训练时遇到的序列更长的序列。

**最终输入 = 词嵌入 (Word Embedding) + 位置编码 (Positional Encoding)**

详情请看： [多头注意力、自注意力与位置编码笔记](../Basic Concepts-NLP/多头注意力、自注意力与位置编码笔记.md)

### 3. 编码器 (Encoder)

每个编码器层包含两个主要的子层：

#### a. 多头自注意力机制 (Multi-Head Self-Attention)

这是 Transformer 的核心创新之一。**自注意力 (Self-Attention)** 允许模型在处理一个词时，关注输入序列中的所有其他词，并根据相关性计算该词的表示。

**i. 缩放点积注意力 (Scaled Dot-Product Attention)**

这是自注意力的基础。对于输入序列中的每个词，我们创建三个向量：查询 (Query, Q)、键 (Key, K) 和值 (Value, V)。这些向量是通过将词的嵌入（加上位置编码）乘以三个不同的可学习权重矩阵 ($W^Q, W^K, W^V$) 得到的。

*   $Q = X W^Q$
*   $K = X W^K$
*   $V = X W^V$
    (其中 X 是输入嵌入矩阵)

注意力得分是通过计算查询向量 Q 与所有键向量 K 的点积得到的。为了防止点积结果过大导致梯度消失，需要将其除以一个 **缩放因子** $\sqrt{d_k}$，其中 $d_k$ 是键向量的维度（通常是 $d_{model} / h$，h 是头的数量）。然后，通过 **Softmax** 函数将得分转换为概率（权重），表示每个词对当前词的重要性。最后，将这些权重乘以对应的值向量 V 并求和，得到该词的注意力输出。

**公式:**
$Attention(Q, K, V) = \text{softmax}\left( \frac{Q K^T}{\sqrt{d_k}} \right) V$

**ii. 多头注意力 (Multi-Head Attention)**

为了让模型能够关注来自不同表示子空间的信息，Transformer 使用了 **多头注意力**。它不是只计算一次注意力，而是将 Q, K, V 通过不同的、可学习的线性投影（权重矩阵 $W_i^Q, W_i^K, W_i^V$）投影 h 次（h 是头的数量，论文中 h=8）。对每个投影后的 Q, K, V 并行地执行缩放点积注意力计算，得到 h 个输出。

$head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)$

然后，将这 h 个输出拼接 (Concatenate) 起来，并通过另一个可学习的线性投影（权重矩阵 $W^O$）得到最终的多头注意力输出。

**公式:**
$MultiHead(Q, K, V) = \text{Concat}(head_1, ..., head_h) W^O$

多头机制允许模型在不同位置共同关注来自不同表示子空间的信息。

#### b. 位置前馈网络 (Position-wise Feed-Forward Network, FFN)

这是编码器层的第二个子层。它是一个简单的、全连接的前馈网络，独立地应用于每个位置（即序列中的每个词）。它包含两个线性变换和一个 ReLU 激活函数。

**公式:**
$FFN(x) = \max(0, x W_1 + b_1) W_2 + b_2$

其中 x 是前一个子层（多头注意力）的输出，$W_1, b_1, W_2, b_2$ 是可学习的参数。这个网络的输入和输出维度都是 $d_{model}$，中间层的维度 $d_{ff}$ 通常更大（论文中为 2048）。

#### c. 残差连接与层归一化 (Add & Norm)

在每个子层（多头注意力和 FFN）的周围，都使用了 **残差连接 (Residual Connection)**，然后进行 **层归一化 (Layer Normalization)**。

*   **残差连接**: 将子层的输入 x 直接加到子层的输出 Sublayer(x) 上，即 $x + \text{Sublayer}(x)$。这有助于缓解深度网络中的梯度消失问题，使得训练更深的模型成为可能。
*   **层归一化**: 对每个样本（在这里是序列中的每个位置）的特征进行归一化，使其均值为 0，方差为 1，然后再进行缩放和平移。这有助于稳定训练过程，加速收敛。

**结构**: $LayerNorm(x + \text{Sublayer}(x))$

所以，一个完整的编码器层流程是：
1.  输入 x
2.  多头自注意力: $attn\_output = \text{MultiHeadSelfAttention}(x)$
3.  Add & Norm: $norm1\_output = LayerNorm(x + attn\_output)$
4.  前馈网络: $ffn\_output = FFN(norm1\_output)$
5.  Add & Norm: $output = LayerNorm(norm1\_output + ffn\_output)$

这个 output 就是该编码器层的最终输出，并作为下一层的输入。

### 4. 解码器 (Decoder)

解码器与编码器结构类似，但有几个关键区别，使其能够生成目标序列。每个解码器层包含三个子层：

#### a. 掩码多头自注意力 (Masked Multi-Head Self-Attention)

解码器的第一个子层是 **掩码 (Masked)** 的多头自注意力。它与编码器的自注意力类似，但是增加了一个 **掩码 (Masking)** 步骤。

**目的**: 在生成目标序列的第 t 个词时，解码器只能关注到位置 t 之前的词（包括位置 t 本身），不能看到未来的词。这是为了确保在预测时，模型的预测只依赖于已生成的输出。

**实现**: 在计算 Scaled Dot-Product Attention 时，在 Softmax 操作之前，将所有对应于未来位置的注意力得分设置为一个非常小的负数（例如 $-\infty$）。这样，经过 Softmax 后，这些位置的权重就接近于 0。

$Attention(Q, K, V) = \text{softmax}\left( \frac{Q K^T}{\sqrt{d_k}} + \text{Mask} \right) V$
(这里的 Mask 是一个上三角矩阵，对角线以上为负无穷，对角线及以下为 0)

#### b. 编码器-解码器注意力 (Encoder-Decoder Attention)

解码器的第二个子层是 **编码器-解码器注意力**，这也是一个多头注意力机制。

**目的**: 允许解码器的每个位置关注编码器输出（即输入序列的表示）中的所有位置。这使得解码器能够将注意力集中在输入序列中与当前预测最相关的部分。

**实现**:

*   **查询 (Query, Q)**: 来自前一个解码器子层（掩码自注意力层）的输出。
*   **键 (Key, K) 和 值 (Value, V)**: 来自 **编码器栈顶** 的输出。

这样，解码器就可以根据它当前的状态（Q）去查询输入序列的信息（K, V）。

**公式:**
$EncoderDecoderAttention(Q_{decoder}, K_{encoder}, V_{encoder})$
$MultiHead(Q_{decoder}, K_{encoder}, V_{encoder}) = \text{Concat}(head_1, ..., head_h) W^O$
$head_i = Attention(Q_{decoder} W_i^Q, K_{encoder} W_i^K, V_{encoder} W_i^V)$

#### c. 位置前馈网络 (Position-wise Feed-Forward Network)

与编码器中的 FFN 完全相同，应用于编码器-解码器注意力子层的输出。

#### d. 残差连接与层归一化 (Add & Norm)

同样，在解码器的每个子层（掩码自注意力、编码器-解码器注意力、FFN）周围，也都使用了残差连接和层归一化。

所以，一个完整的解码器层流程是：
1.  输入 y (来自上一解码器层或目标嵌入+位置编码)
2.  掩码多头自注意力: $masked\_attn\_output = \text{MaskedMultiHeadSelfAttention}(y)$
3.  Add & Norm: $norm1\_output = LayerNorm(y + masked\_attn\_output)$
4.  编码器-解码器注意力: $enc\_dec\_attn\_output = EncoderDecoderAttention(norm1\_output, encoder\_output)$
5.  Add & Norm: $norm2\_output = LayerNorm(norm1\_output + enc\_dec\_attn\_output)$
6.  前馈网络: $ffn\_output = FFN(norm2\_output)$
7.  Add & Norm: $output = LayerNorm(norm2\_output + ffn\_output)$

这个 output 就是该解码器层的最终输出。

### 5. 最终输出层 (Final Output Layer)

解码器栈的最终输出是一系列向量。为了将其转换为每个词的概率，需要经过最后两个步骤：

1.  **线性层 (Linear Layer)**: 一个简单的全连接层，将解码器输出的向量投影到词汇表的大小。输出维度为 V (词汇表大小)。
2.  **Softmax 层**: 将线性层的输出转换为概率分布，表示词汇表中每个词是下一个词的概率。

### 6. 训练 (Training)

*   **损失函数**: 通常使用交叉熵损失 (Cross-Entropy Loss) 来比较模型预测的概率分布和真实的目标词（One-hot 编码）。
*   **优化器**: 论文中使用了 Adam 优化器，并配合特定的学习率调度策略（先线性增加，然后按平方根倒数衰减）。
*   **正则化**: 使用了 Dropout 和 Label Smoothing。

### 总结

Transformer 的核心优势在于：

1.  **并行计算**: 与 RNN 不同，Transformer 中的计算（尤其是自注意力）可以在序列维度上高度并行化，大大加快了训练速度。
2.  **长距离依赖**: 自注意力机制直接计算序列中任意两个位置之间的依赖关系，路径长度为 O(1)，有效解决了 RNN 中的长距离依赖问题。
3.  **模型性能**: 在机器翻译等多种 NLP 任务上取得了当时的最佳效果。

其关键组件包括：
*   输入嵌入 + 位置编码
*   多头自注意力机制（编码器和解码器）
*   掩码多头自注意力机制（解码器）
*   编码器-解码器注意力机制（解码器）
*   位置前馈网络
*   残差连接和层归一化
