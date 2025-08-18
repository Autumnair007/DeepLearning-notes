#  UPerNet笔记

UPerNet（**U**nified **Per**ceptual **Net**work）最初由 Xiao 等人在 ECCV 2018 提出，用于**统一感知解析（Unified Perceptual Parsing, UPP）\**任务——即希望一个模型能在同一张图上同时识别多种视觉概念（场景类别、物体、物体部件、材质、纹理等）。为实现这一目标，UPerNet 提出了一个\**解码器/融合模块**：在任意主干（backbone）之后，把 **Pyramid Pooling Module (PPM)** 与 **Feature Pyramid Network (FPN)** 结合起来，既能获得全局上下文也能进行多尺度特征融合，从而适配多粒度的解析任务。([people.csail.mit.edu](https://people.csail.mit.edu/bzhou/publication/eccv18-segment.pdf?utm_source=chatgpt.com), [openaccess.thecvf.com](https://openaccess.thecvf.com/content_ECCV_2018/html/Tete_Xiao_Unified_Perceptual_Parsing_ECCV_2018_paper.html?utm_source=chatgpt.com))

------

# 总体结构（High-level）

UPerNet 的核心思想很“工程化”：

1. 任意主干（如 ResNet / ResNeXt / 后来的 Transformer/ConvNeXt）提取多层语义特征（常记作 C1,C2,C3,C4C_1, C_2, C_3, C_4，分辨率逐层降低、语义逐层增强）。
2. 在最深层（语义最强、分辨率最低）加入 **PPM** 来补充全局上下文信息。
3. 将 PPM 的输出放入 **FPN 的 top-down 分支**，通过上采样与浅层特征逐层融合得到一组融合后的金字塔特征（P1,P2,…P_1, P_2, \dots）。
4. 对这些融合后的多尺度特征做卷积 / 拼接 / 上采样并送入相应的任务头（task-specific heads），可同时输出像素级分割、部分分割、材质分类或图像级场景分类等。([openaccess.thecvf.com](https://openaccess.thecvf.com/content_ECCV_2018/html/Tete_Xiao_Unified_Perceptual_Parsing_ECCV_2018_paper.html?utm_source=chatgpt.com), [Hugging Face](https://huggingface.co/openmmlab/upernet-convnext-base?utm_source=chatgpt.com))

（实践中很多实现把 PPM 作为 FPN 的“额外输入”——先做全局池化得到的上下文再回填到 top-down 中，提高上下文感知能力。）([openaccess.thecvf.com](https://openaccess.thecvf.com/content_ECCV_2018/html/Tete_Xiao_Unified_Perceptual_Parsing_ECCV_2018_paper.html?utm_source=chatgpt.com))

------

# 关键模块详解（逐个拆开讲清楚）

## 1) Backbone（主干）

- 输出多层特征：记为 C1,C2,C3,C4C_1,C_2,C_3,C_4。通常 C1C_1 分辨率最大、语义最弱，C4C_4 分辨率最小、语义最强。
- UPerNet 对主干没有强绑定，可以插入 ResNet、ResNeXt、ConvNeXt、Swin/ViT 等（很多开源实现与预训练模型也都支持换背骨）。([GitHub](https://github.com/CSAILVision/unifiedparsing?utm_source=chatgpt.com), [Hugging Face](https://huggingface.co/openmmlab/upernet-convnext-base?utm_source=chatgpt.com))

## 2) Pyramid Pooling Module（PPM）

PPM 最早来自 PSPNet 的设计，目的是提供不同尺度的全局上下文。对输入特征 F∈RC×H×WF \in \mathbb{R}^{C\times H \times W}，PPM 做多个不同大小的平均池化，然后通过 1×11\times1 卷积降维并上采样回原分辨率，最后将这些尺度的结果与原特征拼接（concat）并再做一次卷积融合：

PPM(F)=Conv([F, Upsample(P1(F)), Upsample(P2(F)),…,PK(F)])\text{PPM}(F) = \mathrm{Conv}\Big([F,\ \mathrm{Upsample}(P_1(F)),\ \mathrm{Upsample}(P_2(F)),\dots,P_K(F)]\Big)

其中 Pk(F)=Conv(AvgPoolsk(F))P_k(F)=\text{Conv}(\text{AvgPool}_{s_k}(F))，sks_k 为第 kk 个池化尺度（例如 {1,2,3,6}\{1,2,3,6\} 等）。
 解释：平均池化把整个特征“压扁”到较小空间，从而编码更大的接收野（甚至全局），然后再把这些尺度信息带回像素级特征。([openaccess.thecvf.com](https://openaccess.thecvf.com/content_ECCV_2018/html/Tete_Xiao_Unified_Perceptual_Parsing_ECCV_2018_paper.html?utm_source=chatgpt.com))

## 3) Feature Pyramid Network（FPN）——Top-down + lateral

FPN 用来把不同分辨率的特征融合，常见操作是“自顶向下上采样 + 与对应浅层特征逐通道相加（或拼接） + 卷积”，公式化表示：

Pi=Conv(Ci+Upsample(Pi+1))P_i = \mathrm{Conv}\big(C_i + \mathrm{Upsample}(P_{i+1})\big)

其中 CiC_i 是 backbone 的第 ii 层特征，Pi+1P_{i+1} 是上一层融合后的特征（语义更强但分辨率更低），Upsample(⋅)\mathrm{Upsample}(\cdot) 通常为双线性插值或反卷积。
 在 UPerNet 中，作者把 PPM 的输出作为 FPN 顶端的输入（即先把最深层做 PPM，再送入 top-down）——这一步把全局信息注入到整个金字塔的融合过程，使得每一尺度的 PiP_i 都带有全局上下文。([openaccess.thecvf.com](https://openaccess.thecvf.com/content_ECCV_2018/html/Tete_Xiao_Unified_Perceptual_Parsing_ECCV_2018_paper.html?utm_source=chatgpt.com))

## 4) Heads（任务头）

- **像素级分割头**：把多尺度 PiP_i 做上采样到相同分辨率后拼接（或先做逐尺度卷积再拼接），然后用一个 3×33\times3 或 1×11\times1 卷积 + softmax 得到每像素分类概率。
- **图像级场景头**：作者把场景分类（image-level）放在 PPM 之后直接接一个全局池化 + FC，因为场景标签是全图级别信息，放在 PPM 处最合适。
- **多任务设计**：不同任务（parts, materials, objects）可以有独立的输出头，训练时按任务分别计算损失并加权组合。([people.csail.mit.edu](https://people.csail.mit.edu/bzhou/publication/eccv18-segment.pdf?utm_source=chatgpt.com))

------

# 关键数学与损失（带公式并解释）

## 1）PPM 的数学（更紧凑）

设 F∈RC×H×WF\in\mathbb{R}^{C\times H\times W}。对尺度集合 S={s1,…,sK}\mathcal{S}=\{s_1,\dots,s_K\}，定义：

Qk=Upsample(Conv1×1(AvgPoolsk(F))),k=1…K.Q_k = \mathrm{Upsample}\big(\mathrm{Conv}_{1\times1}(\mathrm{AvgPool}_{s_k}(F))\big), \quad k=1\ldots K.

拼接并融合：

Fppm=Conv3×3([F,Q1,Q2,…,QK]).F_{\text{ppm}} = \mathrm{Conv}_{3\times3}\big([F, Q_1, Q_2, \dots, Q_K]\big).

解释：每个 QkQ_k 编码不同尺度的上下文，上采样使它们与 FF 对齐，拼接后再卷积使网络学会如何利用这些上下文信息。

## 2）FPN 融合公式

从顶端开始（假设顶端是 P4P_4）：

P4=Conv(C~4),C~4=PPM(C4),P_4 = \mathrm{Conv}( \tilde C_4 ),\quad \tilde C_4=\text{PPM}(C_4),Pi=Conv(Ci+Upsample(Pi+1)),i=3,2,1.P_i = \mathrm{Conv}\big( C_i + \mathrm{Upsample}(P_{i+1})\big),\quad i=3,2,1.

其中 "+" 表示逐通道相加（也可改为拼接）。Conv\mathrm{Conv} 常是 3×33\times3 卷积用于融合并减小别名效应。

## 3）分割概率与交叉熵

输出 logits zi,j,cz_{i,j,c}（像素 (i,j)(i,j) 在类 cc 的得分），softmax：

pi,j,c=exp⁡(zi,j,c)∑c′exp⁡(zi,j,c′).p_{i,j,c}=\frac{\exp(z_{i,j,c})}{\sum_{c'}\exp(z_{i,j,c'})}.

像素级交叉熵损失：

Lseg=−1N∑i,jlog⁡pi,j,yi,j,L_{\text{seg}} = -\frac{1}{N}\sum_{i,j}\log p_{i,j,y_{i,j}},

其中 yi,jy_{i,j} 为真实类别，NN 为像素数。对于多任务，通常按任务加权：

Ltotal=∑tλtLt,L_{\text{total}}=\sum_{t}\lambda_t L_t,

例如 t∈{scene,object,part,material}t\in\{\text{scene},\text{object},\text{part},\text{material}\}，λt\lambda_t 是权重。UPerNet 在论文中设计了从**异构注释来源**学习的训练策略（下节详述）。([people.csail.mit.edu](https://people.csail.mit.edu/bzhou/publication/eccv18-segment.pdf?utm_source=chatgpt.com))

------

# 训练策略（异构数据与多任务）

UPerNet 的一个创新点不是只是解码器结构，而是**如何用不同数据源（有像素级分割的、有图像级标签的、有局部部件标注的）来联合训练**：

- 对于像素级标注的样本（例如 ADE20K），按像素计算分割损失。
- 对于图像级标注（例如场景标签），在 PPM 后接图像级头并只计算 scene loss（不要求像素标签）。
- 对于只含部分标注（如只标注材料/纹理），对应任务头才计算损失。
   训练时用任务掩码和样本采样策略来避免把不存在的标签强行参与计算，从而在一个模型中学习多种标签类型，达到“统一解析”的目标。([people.csail.mit.edu](https://people.csail.mit.edu/bzhou/publication/eccv18-segment.pdf?utm_source=chatgpt.com))

------

# 为什么 PPM + FPN 的组合有效？（直观 + 数学视角）

- **全局上下文**：PPM 通过大尺度池化把远处像素的信息压缩进来，解决了卷积核受限的接收野问题（有助于区分全局依赖的类别，如“客厅 vs 卧室”或材质判别）。
- **多尺度细节**：FPN 把不同分辨率的特征融合，低层保留高分辨率细节（边界、细小结构），高层保留语义信息（类别、语义一致性）。
- **互补性**：PPM 给 FPN 注入全局语义，FPN 把这些语义经过 top-down 传播到高分辨率分支，使像素预测既有局部细节也受全局约束。数学上，融合操作等价于在不同频带上做加权组合，让下游分类器能同时访问低频（全局）与高频（细节）信息。([openaccess.thecvf.com](https://openaccess.thecvf.com/content_ECCV_2018/html/Tete_Xiao_Unified_Perceptual_Parsing_ECCV_2018_paper.html?utm_source=chatgpt.com))

------

# 实践要点、常见变体与实现注意

- **主干选择**：ResNet 家族很常见；近年来很多工作用 Swin/ViT/ConvNeXt 作为 backbone，再把 UPerNet 的 FPN+PPM 作为 decoder（开源实现广泛存在）。([Hugging Face](https://huggingface.co/openmmlab/upernet-convnext-base?utm_source=chatgpt.com))
- **上采样方式**：双线性插值足够且效率高；在高精度要求下可用反卷积或 learnable upsampling。
- **通道数控制**：PPM 拼接后通道数会膨胀，通常用 1×11\times1 卷积降维以节省计算。
- **归一化与 batch size**：语义分割常受 batch size 限制，建议用 SyncBN 或 GroupNorm 在多卡训练下稳定训练。
- **损失权重调节**：多任务时 λt\lambda_t 的设定会显著影响性能，可用不同比例或动态权重（例如 uncertainty weighting）来平衡。
- **开源实现**：作者与社区有 PyTorch 实现与预训练模型（GitHub、OpenMMLab / HuggingFace 等都提供 UPerNet 的实现/权重，方便替换不同 backbone）。([GitHub](https://github.com/CSAILVision/unifiedparsing?utm_source=chatgpt.com), [Hugging Face](https://huggingface.co/openmmlab/upernet-convnext-base?utm_source=chatgpt.com))

------

# 优缺点（何时用它）

**优点**

- 同时获得全局上下文与高分辨率细节，表现稳定且易于集成多任务头。
- 架构模块化，容易与任意 backbone 配合，工程化强（在工业界被广泛采用作为 decoder baseline）。([Hugging Face](https://huggingface.co/openmmlab/upernet-convnext-base?utm_source=chatgpt.com))

**缺点**

- PPM 拼接会增加通道和计算；FPN 上下采样与多尺度卷积也带来开销。对于极限效率场景（移动端、低算力）可能需要轻量化。
- 多任务训练需要精心设计数据混合与损失权重，否则某些任务会主导训练。

------

# 小结（一句话概括）

UPerNet 的核心贡献是把 **全局上下文（PPM）** 与 **多尺度融合（FPN）** 有机结合，并配合合理的多任务训练策略，使单一模型能在多粒度视觉解析任务上取得良好效果——既有理论可解释性，又有工程可操作性，因此成为语义分割/场景解析的重要基线之一。([people.csail.mit.edu](https://people.csail.mit.edu/bzhou/publication/eccv18-segment.pdf?utm_source=chatgpt.com), [openaccess.thecvf.com](https://openaccess.thecvf.com/content_ECCV_2018/html/Tete_Xiao_Unified_Perceptual_Parsing_ECCV_2018_paper.html?utm_source=chatgpt.com))

------

# 参考

1. Xiao et al., *Unified Perceptual Parsing for Scene Understanding*（論文 & PDF）。([people.csail.mit.edu](https://people.csail.mit.edu/bzhou/publication/eccv18-segment.pdf?utm_source=chatgpt.com))
2. ECCV 2018 会议版本与页面（论文发表页）。([openaccess.thecvf.com](https://openaccess.thecvf.com/content_ECCV_2018/html/Tete_Xiao_Unified_Perceptual_Parsing_ECCV_2018_paper.html?utm_source=chatgpt.com))
3. 作者/社区实现与预训练（GitHub）。([GitHub](https://github.com/CSAILVision/unifiedparsing?utm_source=chatgpt.com))
4. OpenMMLab / HuggingFace 上的 UPerNet 说明（关于与各种 backbones 的集成示例）。([Hugging Face](https://huggingface.co/openmmlab/upernet-convnext-base?utm_source=chatgpt.com))

