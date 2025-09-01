# Segformer 实验增强版 V3

### **实验流程分析与总结**

你所执行的整个 V2 流程，从“激进策略”的失败到“集大成者”的成功，是一个非常宝贵的实践案例。它完美地展示了从“知道要用什么”到“知道该怎么用”的进阶过程。

1.  **科学的迭代闭环**: 你完整地走了一个 `提出假设 -> 实验 -> 分析失败 -> 修正方案 -> 再次实验 -> 获得提升` 的闭环。这是解决复杂问题的最有效路径。特别是从 V2 初版的失败中吸取教训，定位到 `gamma` 参数问题的分析，非常精准。

2.  **对“提升幅度小”的看法**: 首先，我要强调，在像 PASCAL VOC 这样成熟的数据集上，当模型达到一定瓶颈后，mIoU 从 58.6% 提升到 59.33% **并不算是一个微小的进步**。这 0.73 个点的提升是你在正确分析问题后，通过引入更复杂的混合损失函数和数据增强策略“硬啃”下来的，含金量很高。在学术界，很多论文的改进也就集中在 1-2 个百分点。所以，请先肯定自己的成果！

3.  **成功的核心要素**:
    *   **混合损失函数是灵魂**: 你最终的成功，关键在于从单一的 `FocalLoss` 转向了 `FocalLoss` + `DiceLoss` 的组合。这体现了一个深刻的理解：
        *   `FocalLoss` 解决了**像素层面**的类别不平衡（易分/难分样本）。
        *   `DiceLoss` 解决了**区域/结构层面**的类别不平衡（关注预测掩码与真实掩码的重叠度，对小物体更友好）。
        *   二者结合，让模型既能看清“树木”（像素细节），又能看见“森林”（物体结构），是 1+1 > 2 的典范。
    *   **数据增强的协同作用**: `PhotoMetricDistortion` 在一个鲁棒的损失函数（混合损失）的指导下，终于发挥了它应有的正面作用——增强模型泛化能力，而不是加速梯度消失。

**总结**: 你的 V2 实验是一次高质量的优化实践。你不仅提升了模型的性能，更重要的是，你通过亲手实践，深刻理解了高级算法（如 Focal Loss）的超参数敏感性以及不同损失函数之间的协同机制。

---

### **下一步改进方案建议 (V3 探索方向)**

现在，模型性能已经站上了一个新的台阶。要继续突破，我们需要从更多维度进行精细化调优和“压榨”模型的潜力。以下是一些你可以尝试的方向，我将它们从**低成本、易于尝试**到**高成本、更复杂**的顺序列出：

#### **方向一：超参数与训练策略微调 (低成本)**

你的 V2 最终版配置已经非常完善，但仍有微调空间。这些改动不需要大的代码重写，只需修改配置文件。

1. **调整学习率 (Learning Rate)**:

   *   **尝试更低的学习率**: 你的 `learning_rate` 是 `6e-5`。既然模型已经比较成熟，可以尝试用当前最好的模型 (`best_mIoU_epoch_200.pth`) 作为预训练权重，进行一个“微调（fine-tuning）”阶段的训练。在这个阶段，使用一个更小的学习率，例如 `1e-5` 或 `6e-6`，再训练 50-100 个 epochs。这有助于模型在当前的位置进行更精细的搜索，可能会找到一个更好的局部最优点。
   *   **调整学习率调度器**: 你使用了 `LinearLR` (warmup) + `PolyLR`。这是一个非常标准且强大的组合。可以尝试延长 `warmup_epochs` 到 `20` 或 `25`，让模型在初始阶段有更平滑的启动过程，尤其是在使用了复杂数据增强和损失函数时。

   下面是具体的操作步骤和出现的问题：

   ***

   ### 为什么修改的参数不会生效？

   当你使用 `--resume` 标志时，`mmengine` 不仅仅是加载了模型的权重 (`state_dict`)，它还会加载一个完整的训练状态快照，其中包含：

   1.  **模型权重**：这部分当然会加载。
   2.  **优化器状态 (Optimizer State)**：对于像 AdamW 这样的优化器，它会保存每个参数的动量（momentums）和方差（variances）。这些状态是基于*旧的*学习率计算的。
   3.  **学习率调度器状态 (Scheduler State)**：这是最关键的一点。调度器会记录它自己已经执行到了哪一步。在你的情况中，它记录着：“我已经完成了 20 个 epoch 的 warmup，并且已经执行 `PolyLR` 策略到了第 200 个 epoch”。
   4.  **当前的 Epoch 和 Iteration 数**：日志明确显示 `resumed epoch: 200`。

   因此，当你恢复训练时，会发生以下情况：

   *   **`max_epochs = 400`**: 这个参数**会生效**。训练循环会从 epoch 200 开始，一直跑到 400。
   *   **`learning_rate = 0.000006`**: 这个新的基础学习率**不会生效**。优化器和调度器会从检查点中恢复它们的状态，继续使用基于*旧的学习率（0.00006）* 计算出的衰减曲线。
   *   **`warmup_epochs = 20`** 和 **`start_factor=1e-7`**: 这些参数**完全不会生效**。因为恢复的 epoch 是 200，已经远远超过了 warmup 阶段（0-20 epochs）。调度器会直接跳过 `LinearLR` 部分，继续执行它在第 200 个 epoch 时的 `PolyLR` 状态。

   **总结一下：`--resume` 的设计哲学是“精确地从中断的地方继续”，它会忽略掉那些可能与已保存状态冲突的新配置。**

   ### 如果想让新参数生效，应该怎么做？

   如果你想在一个已经训练好的模型基础上，用*全新的*超参数（比如更小的学习率）来微调（fine-tune）模型，你不应该使用 `--resume`。

   你应该使用 `load_from`。

   **解决方案：**

   1. **修改配置文件**:

      *   确保你的配置文件中设置了所有**新**的超参数（`learning_rate`, `max_epochs`, `warmup_epochs`, `param_scheduler` 等）。
      *   在配置文件中，添加 `load_from` 字段，指向你的预训练模型。
      *   **删除或注释掉**模型 backbone 中的 `init_cfg`，因为 `load_from` 会在更高层级上加载权重，避免冲突。

      你的配置文件应该看起来像这样：

      ```python name=my_segformer_mit-b2_3xb6-400e_finetune.py
      # =========================================================================
      #
      #        SegFormer-MiT-B2 在 PASCAL VOC 2012 增强数据集上的
      #       训练配置文件 (v3 - 修改超参数进行微调训练)
      #
      # 作者: Autumnair007
      # 日期: 2025-09-01 (修改学习率，warmup和参数调度器的超参数进行微调训练)
      #
      # =========================================================================
      
      # --- 第 1 部分: 继承基础配置 (只继承模型和运行时) ---
      _base_ = [
          '../_base_/models/segformer_mit-b0.py',
          '../_base_/default_runtime.py'
      ]
      
      # --- 第 2 部分: 硬件与训练超参数 ---
      gpu_count = 3
      samples_per_gpu = 6
      num_workers = 8
      learning_rate = 0.000006 # 新的学习率
      checkpoint_epoch = 10
      val_epoch = 10
      max_epochs = 200
      warmup_epochs = 20 # 新的 warmup
      
      # --- 第 3 部分: 数据集配置 ---
      dataset_type = 'PascalVOCDataset'
      data_root = 'data/VOCdevkit/VOC2012'
      crop_size = (512, 512)
      
      # 训练数据处理流水线 (包含光度畸变)
      train_pipeline = [
          dict(type='LoadImageFromFile'),
          dict(type='LoadAnnotations'),
          dict(
              type='RandomResize',
              scale=(2048, 512),
              ratio_range=(0.5, 2.0),
              keep_ratio=True),
          dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
          dict(type='RandomFlip', prob=0.5),
          dict(type='PhotoMetricDistortion'),  # 光度畸变
          dict(type='Pad', size=crop_size),
          dict(type='PackSegInputs')
      ]
      
      # 测试数据处理流水线
      test_pipeline = [
          dict(type='LoadImageFromFile'),
          dict(type='Resize', scale=(2048, 512), keep_ratio=True),
          dict(type='LoadAnnotations'),
          dict(type='PackSegInputs')
      ]
      
      # 定义训练数据集
      dataset_train = dict(
          type=dataset_type,
          data_root=data_root,
          data_prefix=dict(img_path='JPEGImages', seg_map_path='SegmentationClass'),
          ann_file='ImageSets/Segmentation/train.txt',
          pipeline=train_pipeline)
      
      # 定义增强数据集
      dataset_aug = dict(
          type=dataset_type,
          data_root=data_root,
          data_prefix=dict(
              img_path='JPEGImages', seg_map_path='SegmentationClassAug'),
          ann_file='ImageSets/Segmentation/aug.txt',
          pipeline=train_pipeline)
      
      # --- 第 4 部分: 数据加载器配置 ---
      train_dataloader = dict(
          batch_size=samples_per_gpu,
          num_workers=num_workers,
          persistent_workers=True,
          # 将 InfiniteSampler 换回 DefaultSampler 以支持按 Epoch 训练
          sampler=dict(type='DefaultSampler', shuffle=True),
          dataset=dict(type='ConcatDataset', datasets=[dataset_train, dataset_aug]))
      
      val_dataloader = dict(
          batch_size=1,
          num_workers=num_workers,
          persistent_workers=True,
          sampler=dict(type='DefaultSampler', shuffle=False),
          dataset=dict(
              type=dataset_type,
              data_root=data_root,
              data_prefix=dict(
                  img_path='JPEGImages', seg_map_path='SegmentationClass'),
              ann_file='ImageSets/Segmentation/val.txt',
              pipeline=test_pipeline))
      
      test_dataloader = val_dataloader
      
      # 评估器配置
      val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
      test_evaluator = val_evaluator
      
      # --- 第 5 部分: 模型配置 ---
      data_preprocessor = dict(size=crop_size)
      # 不再需要这个，因为 load_from 会处理
      # checkpoint = 'checkpoints/mit_b2_converted_from_hf.pth' 
      
      model = dict(
          data_preprocessor=data_preprocessor,
          backbone=dict(
              # 【重要】注释或删除这里的 init_cfg，避免和 load_from 冲突
              # init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
              embed_dims=64,
              num_layers=[3, 4, 6, 3]),
          decode_head=dict(
              in_channels=[64, 128, 320, 512],
              num_classes=21,
              loss_decode=[
                  dict(
                      type='FocalLoss',
                      use_sigmoid=True,
                      loss_weight=1.0),
                  dict(
                      type='DiceLoss',
                      loss_weight=1.0,
                      ignore_index=255)
              ]),
      )
      
      # --- 第 6 部分: 优化器与学习率策略 ---
      optimizer = dict(
          type='AdamW', lr=learning_rate, betas=(0.9, 0.999), weight_decay=0.01)
      optim_wrapper = dict(
          type='OptimWrapper',
          optimizer=optimizer,
          paramwise_cfg=dict(
              custom_keys={
                  'pos_block': dict(decay_mult=0.),
                  'norm': dict(decay_mult=0.),
                  'head': dict(lr_mult=10.)
              }))
      
      param_scheduler = [
          dict(
              type='LinearLR',
              start_factor=1e-7, # 从 1e-6 变成 1e-7
              by_epoch=True,
              begin=0,
              end=warmup_epochs,
          ),
          dict(
              type='PolyLR',
              eta_min=0.0,
              power=1.0,
              by_epoch=True,
              begin=warmup_epochs,
              end=max_epochs + 1,
          )
      ]
      
      # --- 第 7 部分: 训练、验证与测试循环配置 ---
      train_cfg = dict(
          type='EpochBasedTrainLoop',
          max_epochs=max_epochs,
          val_interval=val_epoch)
      
      # 【关键】在这里添加 load_from
      load_from = '/home/qz/projects/mmsegmentation/work_dirs/my_segformer_mit-b2_3xb6-200e_voc12aug_v2-advanced-training/epoch_200.pth'
      
      val_cfg = dict(type='ValLoop')
      test_cfg = dict(type='TestLoop')
      
      # --- 第 8 部分: 钩子与可视化配置 ---
      default_hooks = dict(
          timer=dict(type='IterTimerHook'),
          logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=True),
          param_scheduler=dict(type='ParamSchedulerHook'),
          checkpoint=dict(
              type='CheckpointHook',
              by_epoch=True,
              interval=checkpoint_epoch,
              max_keep_ckpts=3,
              save_best='mIoU',
              rule='greater'),
          sampler_seed=dict(type='DistSamplerSeedHook'),
          visualization=dict(type='SegVisualizationHook'))
      
      vis_backends = [
          dict(type='LocalVisBackend'),
          dict(type='TensorboardVisBackend')
      ]
      visualizer = dict(
          type='SegLocalVisualizer',
          vis_backends=vis_backends,
          name='visualizer'
      )
      ```

   2. **修改并运行命令**:

      *   **去掉 `--resume` 标志！**
      *   运行新的训练命令。

      ```bash
      # 注意，没有 --resume
      CUDA_VISIBLE_DEVICES=5,6,7 ./tools/dist_train.sh configs/segformer/my_segformer_mit-b2_3xb6-400e_finetune.py 3d
      ```

   **这样做会发生什么？**

   *   训练会从 **Epoch 1** 开始。
   *   `load_from` 会将 `epoch_200.pth` 文件中模型的**权重**加载到你的新模型中。
   *   **但是**，它不会加载优化器和学习率调度器的状态。
   *   优化器和调度器会根据你配置文件中的**新设置**（`lr=0.000006`, `warmup_epochs=20` 等）进行全新的初始化。
   *   训练会从一个新的、非常低的学习率开始，并按照你新设定的策略（20个epoch的warmup，然后Poly衰减）进行训练，直到新的200个epoch。

   这正是“加载预训练权重并用新参数进行微调”的标准做法。

   ***

   tmux的命令如下：

   * **创建会话**: 

     ```bash
     tmux new-session -s seg_train_b6_v3
     conda activate open-mmlab
     ```

   * **恢复会话：**

     ```bash
     tmux attach -t seg_train_b6_v3
     ```

   * **删除旧会话**: 

     ```bash
     tmux kill-session -t seg_train_b6_v3
     ```

   ***

2. **损失权重调整**:

   *   在你的混合损失中，`FocalLoss` 和 `DiceLoss` 的权重都是 `1.0`。这是一个很好的起点，但未必是最佳组合。你可以尝试不平衡的权重，例如：
       *   `loss_weight` for `FocalLoss`: 1.0, `loss_weight` for `DiceLoss`: **0.5** (更侧重像素精度)
       *   `loss_weight` for `FocalLoss`: 0.5, `loss_weight` for `DiceLoss`: **1.0** (更侧重区域结构)
   *   这需要一些实验来找到最佳配比，但有时能带来意外的惊喜。

#### **方向二：数据增强“军火库”升级 (中等成本)**

你的流水线中已经有了最核心的几何增强（RandomResize, RandomCrop, RandomFlip）和色彩增强（PhotoMetricDistortion）。我们可以引入更多“猛料”。

1.  **CutOut / Random Erasing**:
    *   **是什么**: 在图像上随机挖掉一个或多个矩形区域。
    *   **为什么有效**: 强迫模型利用物体的上下文信息进行预测，而不是仅仅依赖于物体的某个局部特征。这能极大地提高模型的鲁棒性。MMSegmentation 中内置了 `RandomErasing`。
    *   **如何添加**: 在 `train_pipeline` 的 `PhotoMetricDistortion` 之后加入 `dict(type='RandomErasing', prob=0.5, ...)`。你需要根据文档调整其参数。

2.  **MixUp / CutMix**:
    *   **是什么**: 这是两种更高级的增强策略。`MixUp` 将两张图片按比例混合，`CutMix` 则是将一张图的一部分裁切并粘贴到另一张图上。它们的标签也做相应混合。
    *   **为什么有效**: 它们创造了在真实世界中不存在的、更具挑战性的样本，极大地扩展了训练数据的分布，能有效抑制过拟合，提升泛化能力。
    *   **如何添加**: MMSegmentation 提供了 `RandomMix` 和 `RandomCut`。这两种增强通常放在 `PackSegInputs` 之前，因为它们涉及到多张图的操作。注意，这两种增强的计算开销会比 CutOut 更大。

#### **方向三：模型与数据策略的变革 (高成本)**

这些是更根本性的改变，可能需要修改更多代码或进行更复杂的数据处理。

1.  **OHEM (Online Hard Example Mining)**:
    *   **是什么**: 不再平等地对待所有像素，而是在线（每个 mini-batch）地只选择那些损失值最高的像素（即最难分的像素）来回传梯度。
    *   **为什么有效**: 这是解决类别不平衡的另一种经典思路。它能让模型更专注于学习那些模棱两可、难以区分的边界或小物体。
    *   **如何实现**: 在 `decode_head` 的 `loss_decode` 中，除了损失类型，还可以配置 `sampler`。你可以将 `CrossEntropyLoss` 或 `FocalLoss` 与 `OHEMPixelSampler` 结合使用。这通常需要你重写 `loss_decode` 部分，将一个损失函数包裹在 OHEM 采样器逻辑中。

2.  **引入额外数据 (External Data)**:
    *   **是什么**: PASCAL VOC 2012 的 `aug` 数据集（SBD）是一个很好的补充。但你还可以考虑引入 COCO 数据集的部分数据进行预训练。
    *   **为什么有效**: 更多、更多样化的数据是提升模型性能上限最朴素也最有效的方法。在 COCO 上预训练过的模型，其特征提取能力通常会比只在 ImageNet 上预训练的模型更强，尤其对于分割任务。
    *   **如何实现**: 这会是一个更复杂的流程。你需要在 COCO 上先训练你的 Segformer，然后将得到的权重作为预训练模型，再到你的 PASCAL VOC 数据集上进行微调。

### **行动建议**

我建议你按照以下顺序，循序渐进地尝试：

1.  **首先尝试方向一**: 进行学习率微调。这是成本最低、见效可能最快的方法。用你现有的 `59.33%` 的模型作为起点，用小 10 倍的学习率再跑 50 个 epoch 看看。
2.  **如果微调效果不明显，再尝试方向二**: 在你的 V2 最终版配置中，加入 `RandomErasing`。这是一个非常实用的增强技术，与你现有的策略能很好地互补。
3.  **如果想追求极致性能，再考虑方向三**: OHEM 或引入外部数据是冲击更高分数的“大招”，但它们需要更多的时间和精力投入。

祝你的 V3 实验顺利，期待你的模型能再次突破性能记录！



