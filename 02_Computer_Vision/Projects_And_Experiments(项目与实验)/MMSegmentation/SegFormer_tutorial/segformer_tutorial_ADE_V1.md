# Segformer ADE数据集 V1

从PASCAL VOC 2012 增强数据集V3版本继承继续研究。

参考资料：[(一)ADE20K数据集-CSDN博客](https://blog.csdn.net/lx_ros/article/details/125650685)

ADEchallenge 2016 的下载地址为： http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip

官网下载地址为：[ADE20K dataset](https://ade20k.csail.mit.edu/)，要注册。GitHub仓库为：[CSAILVision/ADE20K: ADE20K Dataset](https://github.com/CSAILVision/ADE20K)

***

### 官方下载的数据集和其他地方数据集的差别

简单来说，**你下载的 `ADE20K_2021_17_01` 和你在其他地方看到的 `ADEChallengeData2016`，很可能指向的是同一个核心数据集，即包含150个语义类别、超过2万张训练图像和2千张验证图像的那个版本。** 它们本质内容相同，只是官方在不同时期可能用了不同的打包命名方式。

下面是一个表格，帮你快速了解这两个版本标识的关系和区别：

| 特性维度              | ADEChallengeData2016 (常见于论文和代码库)                    | ADE20K_2021_17_01 (你从官网下载的版本)                       |
| :-------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| **常见名称**          | ADE20K (Scene Parsing), ADE20K-150                           | 可能为数据包内部版本标识或特定发布版名称                     |
| **核心数据内容**      | **150个语义类别**，**20,210张训练图**，**2,000张验证图**     | 应包含相同的150个类别及图像数据（需确认`objects.txt`内容）   |
| **主要用途**          | **语义分割**、场景解析（如SegFormer等模型训练）              | 同左                                                         |
| **来源与引用**        | MIT发布，论文和框架（如MMSegmentation）中常用此名称          | 应源自同一官方渠道，可能为更新或重新打包的版本标识           |
| **文件结构参考**      | `ADEChallengeData2016`<br>  ├── `annotations`<br>  │   ├── `training`<br>  │   └── `validation`<br>  └── `images`<br>      ├── `training`<br>      └── `validation` | `ADE20K_2021_17_01`<br>  ├── `images`<br>  │   └── `ADE`<br>  ├── `training` (子目录按场景分类)<br>  ├── `validation`<br>  ├── `index_ade20k.mat`<br>  ├── `index_ade20k.pkl`<br>  └── `objects.txt` |
| **与SegFormer的关联** | SegFormer论文及MMSegmentation等框架**均使用此版本名称**      | 核心数据应与前者一致，可供SegFormer训练                      |

🧐 **SegFormer论文与MMSegmentation用的是哪个？**

根据你提供的MMSegmentation中SegFormer的README，以及搜索结果，**SegFormer原始论文和MMSegmentation等主流框架中提到的ADE20K数据集，指的就是 `ADEChallengeData2016` 这个版本**（包含150个类别，20210张训练图像和2000张验证图像）。你提供的MMSegmentation的配置文件中也多次出现了 `ADE20K` 的字段，这指向的同样是这个150个类别的数据集。

你从官网下载的 `ADE20K_2021_17_01`，其名称中的“2021”可能表示数据打包或发布的年份，而“17_01”可能是一个内部版本号或批次标识。**关键在于其核心内容（图像和150个类别的标注）应与 `ADEChallengeData2016` 一致。**

💡 **给你的建议**

1.  **确认数据内容**：打开你下载的 `ADE20K_2021_17_01` 中的 `objects.txt` 文件，检查它是否包含了150个类别的名称列表。这是确认它是否是150类别版本的最直接方法。
2.  **用于训练**：如果你想在MMSegmentation等框架中训练SegFormer或其他模型，**核心是确保你的数据路径设置正确**。虽然文件夹名称不同，你只需要在配置文件中将数据路径指向你解压后的 `ADE20K_2021_17_01` 文件夹即可，或者将其重命名为框架期望的名称（如 `ADEChallengeData2016`），但更推荐**修改配置文件中的路径**以避免混淆。
3.  **理解差异**：`ADE20K_2021_17_01` 的 `training` 目录下包含了按场景类型（如cultural, urban等）划分的子目录，这可能是官方提供的另一种数据组织方式，便于用户按场景查找图像，但**最终的语义分割标注（annotations）应该还是那150个类别**。

📌 **总结一下**
你可以认为 `ADEChallengeData2016` 是学术论文和代码框架中**约定俗成的名称**，指向那个经典的150类别数据集。而你下载的 `ADE20K_2021_17_01` 很可能是**官方后续提供的一个数据包版本**，其核心内容与前者一致，只是打包方式和顶层目录结构略有不同。

因此，**不必过于纠结名称差异**，只要你的数据集包含150个类别、超过2万张训练图像和2千张验证图像，就可以放心地把它当作“那个”ADE20K数据集来使用，并用于训练像SegFormer这样的模型。

***

代码的主要修改点如下：

1. **数据集配置**：
   - `dataset_type` 已从 `PascalVOCDataset` 更改为 `ADE20KDataset`。
   - `data_root` 已更新为 `data/ADEChallengeData2016`。
   - `data_prefix` 和 `ann_file` 已根据 `ADE20K` 的目录结构进行了调整。`ADE20K` 不需要 `ann_file`，并且训练/验证集分别在不同的子目录中。
   - 由于 `ADE20K` 没有像 `VOC` 那样的 `aug.txt` 增强集，因此我移除了 `ConcatDataset`，现在只使用标准的训练集。
2. **模型配置**：
   - 解码头（`decode_head`）的 `num_classes` 已从 `21`（PASCAL VOC）更改为 `150`（ADE20K）。
   - `ignore_index` 在 `ADE20K` 中通常是 `255`，这与你的损失函数设置一致，予以保留。
   - 将FocalLoss参数权重下调，提高DiceLoss权重
   - 在训练流水线中，使用 RandomCutOut 替代 RandomErasing。
3. **预训练权重（重要修正）**：
   - `mit_b2_converted_from_hf.pth` 应该作为 **backbone 的预训练权重**，而不是使用 `load_from` 来加载整个模型的 checkpoint。`load_from` 会尝试加载所有组件（包括解码头），这在数据集类别数不同时会引发错误。
   - 因此，我**移除了全局的 `load_from` 配置**。
   - 并在 `model` -> `backbone` 配置中，**重新启用了 `init_cfg`**，并将其 `checkpoint` 路径指向你提供的 `checkpoints/mit_b2_converted_from_hf.pth`。
   - 这样，MMSegmentation 会在初始化模型时，只加载主干网络（Backbone）的权重，而解码头（Decode Head）等其他部分将进行随机初始化，这正是从头开始训练新数据集（如 ADE20K）的标准做法。

其他所有配置，包括学习率、优化器、数据增强流水线 (`train_pipeline`)、训练周期 (`max_epochs`) 等，都保持了你原有的设置。

```python name=my_segformer_mit-b2_512x512_200e_ade20k.py
# =========================================================================
#
#        SegFormer-MiT-B2 在 ADE20K 数据集上的
#       训练配置文件 (v1.0 - 基于 PASCAL VOC v3.1 脚本修改)
#
# 作者: Autumnair007
# 日期: 2025-09-02
#
# =========================================================================

# --- 第 1 部分: 继承基础配置 ---
_base_ = [
    '../_base_/models/segformer_mit-b0.py', # 基础模型结构，后续会覆盖
    '../_base_/default_runtime.py'
]

checkpoint = 'checkpoints/mit_b2_converted_from_hf.pth'

# --- 第 2 部分: 硬件与训练超参数 ---
gpu_count = 3
samples_per_gpu = 2
num_workers = 8
learning_rate = 0.000006
checkpoint_epoch = 10
val_epoch = 10
max_epochs = 200
warmup_epochs = 20

# --- 第 3 部分: 数据集配置 (修改为 ADE20K) ---
dataset_type = 'ADE20KDataset'
data_root = 'data/ADEChallengeData2016' # <-- 修改: 数据集根目录
crop_size = (512, 512)

# 训练数据处理流水线
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(
        type='RandomResize',
        scale=(2048, 512),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    # 使用 RandomCutOut 替代 RandomErasing
    dict(type='RandomCutOut', 
         prob=0.5, 
         n_holes=(1, 3),
         cutout_ratio=[(0.02, 0.02), (0.2, 0.2)],
         fill_in=(0, 0, 0), 
         seg_fill_in=255),
    dict(type='Pad', size=crop_size),
    dict(type='PackSegInputs')
]

# 测试数据处理流水线
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 512), keep_ratio=True),
    dict(type='LoadAnnotations', reduce_zero_label=True), # ADE20K 标签从 1 开始，0 是背景，需要减一
    dict(type='PackSegInputs')
]

# --- 第 4 部分: 数据加载器配置 (修改为 ADE20K) ---
train_dataloader = dict(
    batch_size=samples_per_gpu,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/training', seg_map_path='annotations/training'), # <-- 修改: ADE20K 路径
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/validation', seg_map_path='annotations/validation'), # <-- 修改: ADE20K 路径
        pipeline=test_pipeline))

test_dataloader = val_dataloader

# 评估器配置
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

# --- 第 5 部分: 模型配置 ---
data_preprocessor = dict(
    size=crop_size
)

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        # 保持 MiT-B2 的配置
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        embed_dims=64,
        num_layers=[3, 4, 6, 3]),
    decode_head=dict(
        in_channels=[64, 128, 320, 512],
        num_classes=150, # <-- 修改: ADE20K 类别数为 150
        loss_decode=[
            dict(
                type='FocalLoss',
                use_sigmoid=True,
                loss_weight=0.7),
            dict(
                type='DiceLoss',
                loss_weight=1.3,
                ignore_index=255) # ADE20K 的 ignore_index 也是 255
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
        start_factor=1e-7,
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

### 运行命令：

```bash
CUDA_VISIBLE_DEVICES=5,6,7 ./tools/dist_train.sh configs/segformer/my_segformer_mit-b2_512x512_200e_ade20k.py 3
```

#### tmux的命令如下：

* **创建会话**: 

  ```bash
  tmux new-session -s seg_ade_v1
  conda activate open-mmlab
  ```

* **恢复会话：**

  ```bash
  tmux attach -t seg_ade_v1
  ```

* **删除旧会话**: 

  ```bash
  tmux kill-session -t seg_ade_v1
  ```

#### 评估模型代码：

```bash
CONFIG_FILE="configs/segformer/my_segformer_mit-b2_3xb6-400e_finetune.py"
# 【注意】工作目录名会根据配置文件名自动改变
CHECKPOINT_FILE="work_dirs/my_segformer_mit-b2_3xb6-400e_finetune/best_mIoU_epoch_190.pth"

CUDA_VISIBLE_DEVICES=5 python tools/test.py $CONFIG_FILE $CHECKPOINT_FILE --show-dir outputs/segformer_400e_finetune
```

评估结果如下：
