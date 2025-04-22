#  Transformer代码笔记（Gemini2.5Pro生成）

ps: 自己的电脑用的是CPU，大概10分钟1轮。

代码主要分为以下几个部分：

1.  **导入库**: 引入必要的 Python 库。
2.  **环境设置**: 配置运行环境，如设备（CPU/GPU）、随机种子和数据存储路径。
3.  **数据处理**: 下载、解压、清洗、分词、构建词汇表，并将文本数据转换为模型可处理的格式。
4.  **数据集类**: 定义 `Dataset` 类来组织和提供数据。
5.  **模型定义**: 构建 Transformer 分类器模型。
6.  **数据加载器**: 创建 `DataLoader` 以便在训练和评估时按批次加载数据。
7.  **训练函数**: 定义模型训练逻辑。
8.  **评估函数**: 定义模型评估逻辑（计算损失和准确率）。
9.  **可视化函数**: 定义绘制混淆矩阵和训练历史曲线的函数。
10. **主函数**: 整合所有部分，执行数据加载、模型训练、评估和结果保存。
11. **执行入口**: 确保 `main` 函数在脚本直接运行时被调用。

---

### 1. 导入库 (Imports)

```python
import torch                     # PyTorch 核心库，用于张量操作和神经网络
import torch.nn as nn            # PyTorch 神经网络模块，包含各种层（如线性层、嵌入层）和损失函数
import torch.optim as optim      # PyTorch 优化器模块，包含 Adam 等优化算法
from torch.utils.data import Dataset, DataLoader # PyTorch 数据处理工具，用于创建自定义数据集和数据加载器
import numpy as np               # NumPy 库，用于数值计算（主要用于评估指标）
import matplotlib.pyplot as plt  # Matplotlib 库，用于绘制图表
from sklearn.metrics import accuracy_score, confusion_matrix # Scikit-learn 库，用于计算准确率和混淆矩阵
import seaborn as sns            # Seaborn 库，用于绘制更美观的统计图表（如热力图）
import pandas as pd              # Pandas 库（在此代码中未使用，但常用于数据处理）
import time                      # 时间库，用于计算训练耗时
import re                        # 正则表达式库，用于文本清洗
import os                        # 操作系统接口库，用于文件和目录操作
import tarfile                   # 用于处理 .tar.gz 压缩文件
import urllib.request            # 用于从 URL 下载文件
from collections import Counter  # 用于统计词频，构建词汇表
from torch.nn.utils.rnn import pad_sequence # 用于将不同长度的序列填充到相同长度
from tqdm import tqdm            # 用于显示进度条，方便跟踪长时间运行的任务
```

这部分导入了所有需要的库。每个库都有其特定的用途，例如 PyTorch 用于构建和训练模型，Scikit-learn 用于评估，Matplotlib 和 Seaborn 用于可视化，os 和 re 用于数据处理等。

---

### 2. 环境设置 (Setup)

```python
# 设置数据集下载路径为当前文件夹
current_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
data_dir = os.path.join(current_dir, "data")
os.makedirs(data_dir, exist_ok=True)

# 设置随机种子以便结果可复现
torch.manual_seed(1234)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
print(f"数据集将下载到: {data_dir}")

if torch.cuda.is_available():
    print('CUDA版本:', torch.version.cuda)
```

*   `current_dir` 和 `data_dir`: 确定脚本所在的目录，并在该目录下创建一个名为 `data` 的子目录，用于存放下载的数据集。`os.makedirs(data_dir, exist_ok=True)` 确保目录存在，如果已存在则不报错。
*   `torch.manual_seed(1234)`: 设置 PyTorch 的随机种子。这使得代码每次运行时，随机数生成器的行为都是一样的，有助于保证实验结果的可复现性（例如，模型权重的初始值、Dropout 的行为等）。
*   `device`: 检查是否有可用的 CUDA GPU。如果有，则将 `device` 设置为 'cuda'，否则设置为 'cpu'。后续的张量和模型都会被移动到这个设备上进行计算。
*   打印信息: 输出当前使用的设备（CPU 或 GPU）以及数据集的存储路径。如果使用 GPU，还会打印 CUDA 版本。

---

### 3. 数据处理 (Data Processing)

这部分包含多个函数，用于下载、解压、清洗和准备 IMDB 数据集。

```python
# 手动下载并处理IMDB数据集
def download_and_extract_imdb():
    # ... (代码省略)
    return extracted_path

# 简单的分词器
def basic_english_tokenizer(text):
    # ... (代码省略)
    return text.strip().split()

def clean_text(text):
    # ... (代码省略)
    return text.strip()

# 加载IMDB数据集（不使用torchtext）
def load_imdb_data_manually():
    # ... (代码省略)
    return train_dataset, test_dataset, vocab
```

*   **`download_and_extract_imdb()`**:
    *   定义 IMDB 数据集的下载 URL 和本地存储路径。
    *   检查数据集是否已解压。如果未解压，则检查压缩包是否存在。
    *   如果压缩包不存在，使用 `urllib.request.urlretrieve` 下载。
    *   下载完成后，使用 `tarfile` 解压 `.tar.gz` 文件到 `data_dir`。
    *   返回解压后的数据集根目录路径 (`aclImdb`)。
*   **`clean_text(text)`**:
    *   将文本转换为小写 (`text.lower()`)。
    *   移除 HTML 换行符 `<br />` (`re.sub(r'<br />', ' ', text)`)。
    *   移除除字母、数字和空格外的所有标点符号 (`re.sub(r'[^\w\s]', ' ', text)`)。
    *   将多个连续空格合并为一个 (`re.sub(r'\s+', ' ', text)`)。
    *   移除首尾多余的空格 (`text.strip()`)。
    *   返回清洗后的文本。
*   **`basic_english_tokenizer(text)`**:
    *   这是一个简化的英文分词器。
    *   它先将文本转为小写，移除标点符号（替换为空格），合并多余空格，最后按空格分割文本，返回一个包含单词（tokens）的列表。
*   **`load_imdb_data_manually()`**:
    *   调用 `download_and_extract_imdb()` 获取数据集路径。
    *   定义使用的分词器 (`tokenizer = basic_english_tokenizer`)。
    *   **加载数据**: 遍历 `aclImdb/train/pos`、`aclImdb/train/neg`、`aclImdb/test/pos`、`aclImdb/test/neg` 目录下的 `.txt` 文件。
    *   对每个文件：读取内容，调用 `clean_text()` 清洗，将清洗后的文本和对应的标签（正面评价为 1，负面为 0）分别存入 `train_texts`/`test_texts` 和 `train_labels`/`test_labels` 列表。使用 `tqdm` 显示加载进度。
    *   **构建词汇表 (Vocabulary)**:
        *   使用 `collections.Counter` 统计训练集 (`train_texts`) 中所有单词的出现频率。
        *   创建一个 `vocab` 字典。首先添加特殊标记：`<pad>` (填充标记，索引为 0) 和 `<unk>` (未知词标记，索引为 1)。
        *   遍历 `Counter` 中的词和频率，只将频率大于等于 5 的词加入 `vocab`，并分配递增的索引。频率低的词会被视为 `<unk>`。
        *   打印词汇表大小。
    *   **创建数据集对象**: 使用处理好的文本、标签、词汇表和分词器，创建 `IMDBDataset` 类的实例 (`train_dataset`, `test_dataset`)。
    *   返回训练数据集、测试数据集和词汇表。

---

### 4. 数据集类 (Dataset Class)

```python
class IMDBDataset(Dataset):
    def __init__(self, texts, labels, vocab, tokenizer, max_len=512):
        # ... (代码省略)

    def __len__(self):
        # ... (代码省略)

    def __getitem__(self, idx):
        # ... (代码省略)
        return {
            'text': torch.tensor(token_ids, dtype=torch.long),
            'length': len(token_ids),
            'label': torch.tensor(label, dtype=torch.long)
        }
```

*   继承自 `torch.utils.data.Dataset`，这是 PyTorch 中自定义数据集的标准方式。
*   **`__init__(...)`**: 构造函数，接收文本列表、标签列表、词汇表、分词器和最大序列长度 (`max_len`) 作为输入，并将它们存储为类的属性。
*   **`__len__(self)`**: 返回数据集中样本的总数。`DataLoader` 会使用这个方法来确定数据集的大小。
*   **`__getitem__(self, idx)`**: 根据索引 `idx` 获取单个数据样本。
    *   获取对应索引的文本 (`text`) 和标签 (`label`)。
    *   使用 `self.tokenizer` 将文本分词。
    *   如果分词后的 token 数量超过 `max_len`，则截断，只保留前 `max_len` 个 token。
    *   **文本转索引**: 遍历 tokens，使用 `self.vocab.get(token, self.vocab['<unk>'])` 将每个 token 转换为其在词汇表中的索引。如果 token 不在词汇表中，则使用 `<unk>` 的索引。
    *   将 token 索引列表 (`token_ids`) 和标签 (`label`) 转换为 PyTorch 张量 (`torch.tensor`)。标签类型为 `torch.long`，因为后续的损失函数（如 `CrossEntropyLoss`）需要长整型标签。
    *   返回一个字典，包含处理后的文本张量 (`'text'`)、原始文本长度 (`'length'`) 和标签张量 (`'label'`)。

---

### 5. 模型定义 (Model Definition)

这部分定义了 Transformer 模型的核心组件和最终的分类器。

```python
# 定义Transformer编码器层
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        # ... (代码省略)

    def forward(self, x, mask=None):
        # ... (代码省略)
        return x

# 定义Transformer分类器
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, ff_dim, num_classes, max_len=512, dropout=0.1):
        # ... (代码省略)

    def forward(self, x, lengths):
        # ... (代码省略)
        return logits

def create_padding_mask(batch):
    # ... (代码省略)
```

*   **`TransformerEncoderLayer`**: 定义了 Transformer 模型中的一个编码器层。
    *   **`__init__(...)`**:
        *   `self_attn`: 定义多头自注意力（Multi-head Self-Attention）层 (`nn.MultiheadAttention`)。它接收嵌入维度 `embed_dim`、注意力头数 `num_heads` 和 dropout 率。`embed_dim` 定义了模型中表示每个单词（token）的基础向量维度，它决定了单词表示的丰富程度和模型的基本“宽度”。`num_heads` 将 `embed_dim` 分割开，允许多个注意力机制并行运行在不同的表示子空间上，使模型能够同时关注输入的不同方面和关系。
        *   `feed_forward`: 定义前馈神经网络。它由两个线性层 (`nn.Linear`) 和一个 ReLU 激活函数组成，中间也有 Dropout。维度变化是 `embed_dim -> ff_dim -> embed_dim`。
        *   `norm1`, `norm2`: 定义层归一化（Layer Normalization）层 (`nn.LayerNorm`)。分别用在自注意力和前馈网络之后。
        *   `dropout`: 定义 Dropout 层 (`nn.Dropout`)。
    *   **`forward(self, x, mask=None)`**: 定义前向传播逻辑。
        *   **自注意力**: 输入 `x` 通过多头自注意力层。`mask` 用于指示哪些位置是填充的，注意力机制不应关注这些位置。
        *   **残差连接与归一化 (Add & Norm)**: 将自注意力层的输出通过 Dropout 后与原始输入 `x` 相加（残差连接），然后进行层归一化 (`self.norm1`)。
        *   **前馈网络**: 上一步的输出通过前馈网络。
        *   **残差连接与归一化 (Add & Norm)**: 将前馈网络的输出通过 Dropout 后与前一步的输出相加，然后进行层归一化 (`self.norm2`)。
        *   返回编码器层的最终输出。
*   **`TransformerClassifier`**: 定义了整个分类模型。
    *   **`__init__(...)`**:
        *   `token_embedding`: 词嵌入层 (`nn.Embedding`)。将输入的 token 索引映射为密集向量。`vocab_size` 是词汇表大小，`embed_dim` 是嵌入向量的维度。
        *   `position_embedding`: 位置嵌入 (`nn.Parameter`)。这是一个可学习的参数张量，形状为 `(max_len, embed_dim)`。它为序列中的每个位置提供一个独特的嵌入向量，让模型能够理解单词的顺序。
        *   `transformer_layers`: 使用 `nn.ModuleList` 包含多个 `TransformerEncoderLayer` 实例。`num_layers` 控制编码器的层数。
        *   `classifier`: 最终的线性分类层。将 Transformer 的输出映射到类别得分。`num_classes` 是分类任务的类别数（这里是 2，正面/负面）。
        *   `dropout`: Dropout 层，用于分类器之前，防止过拟合。
    *   **`forward(self, x, lengths)`**: 定义模型的前向传播逻辑。
        *   `x`: 输入的批次数据，形状为 `[batch_size, seq_len]`，包含 token 索引。
        *   `lengths`: 包含批次中每个序列的原始长度（未使用，但保留了接口）。
        *   `mask = create_padding_mask(x)`: 创建填充掩码，标记输入序列中的填充位置（通常是索引为 0 的 `<pad>` token）。
        *   **嵌入**: 输入 `x` 通过 `token_embedding` 层得到词嵌入。
        *   **位置编码**: 将词嵌入与 `position_embedding`（取序列长度部分）相加，得到结合了词义和位置信息的嵌入。
        *   **维度转换**: 使用 `x.transpose(0, 1)` 将形状从 `[batch_size, seq_len, embed_dim]` 转换为 `[seq_len, batch_size, embed_dim]`，以符合 PyTorch `MultiheadAttention` 的输入要求。
        *   **Transformer 编码**: 数据依次通过 `transformer_layers` 中的每个 `TransformerEncoderLayer`。填充掩码 `mask` 会传递给注意力层。
        *   **维度转换**: 使用 `x.transpose(0, 1)` 将形状转回 `[batch_size, seq_len, embed_dim]`。
        *   **池化 (Pooling)**:
            *   获取反转后的掩码 (`1 - mask`)，其中 1 代表有效位置，0 代表填充位置。
            *   将 Transformer 最后一层的输出 `x` 与反转掩码相乘（广播机制），使得填充位置的向量变为 0。
            *   沿着序列长度维度 (`dim=1`) 求和，再除以每个序列的有效长度（`mask.sum(dim=1)`），得到每个序列的平均向量（Mean Pooling）。`.clamp(min=1e-9)` 防止除以零。这是将变长序列表示为固定长度向量的一种常用方法。
        *   **分类**: 将池化后的向量通过 Dropout 层，然后输入到 `classifier` 线性层，得到每个类别的原始得分（logits）。
        *   返回 logits。
*   **`create_padding_mask(batch)`**:
    *   接收一个批次的 token 索引张量 `batch`。
    *   通过 `(batch == 0)` 创建一个布尔张量，其中值为 `True` 的位置对应原始张量中值为 0（即 `<pad>` token）的位置。这个掩码用于告知 `MultiheadAttention` 忽略这些填充位置。

---

### 6. 数据加载器 (DataLoader and Collation)

```python
# 批处理函数：将不同长度的序列填充到相同长度
def collate_batch(batch):
    # ... (代码省略)
    return {
        'texts': padded_texts,
        'labels': torch.stack(labels),
        'lengths': torch.tensor(lengths)
    }

# 在 main 函数中创建 DataLoader
train_dataloader = DataLoader(...)
test_dataloader = DataLoader(...)
```

*   **`collate_batch(batch)`**: 这个函数是传递给 `DataLoader` 的 `collate_fn` 参数。`DataLoader` 在组合单个样本 (`__getitem__` 的返回值) 形成一个批次时会调用它。
    *   接收一个列表 `batch`，其中每个元素都是 `IMDBDataset.__getitem__` 返回的字典。
    *   从批次中提取所有的文本 (`texts`)、标签 (`labels`) 和长度 (`lengths`)。
    *   **填充 (Padding)**: 使用 `pad_sequence(texts, batch_first=True, padding_value=0)`。
        *   `texts` 是一个包含不同长度张量的列表。
        *   `pad_sequence` 会自动计算批次中最长序列的长度。
        *   将所有短于最长序列的张量用 `padding_value=0` (即 `<pad>` token 的索引) 填充到相同的长度。
        *   `batch_first=True` 表示返回的张量形状为 `[batch_size, max_seq_len]`。
    *   使用 `torch.stack(labels)` 将标签列表堆叠成一个张量。
    *   将长度列表转换为张量。
    *   返回一个包含填充后的文本批次、标签批次和长度批次的字典。
*   **`DataLoader(...)`**:
    *   创建数据加载器实例。
    *   `train_subset`/`test_subset`: 要加载的数据集。
    *   `batch_size=32`: 每个批次包含 32 个样本。
    *   `shuffle=True` (用于训练集): 在每个 epoch 开始时打乱数据顺序，有助于模型训练。
    *   `shuffle=False` (用于测试集): 不需要打乱。
    *   `collate_fn=collate_batch`: 指定使用我们定义的 `collate_batch` 函数来组合样本并进行填充。

---

### 7. 训练函数 (Training Function)

```python
def train(model, dataloader, optimizer, criterion, device):
    model.train() # 设置模型为训练模式（启用 Dropout 等）
    total_loss = 0
    progress_bar = tqdm(dataloader, desc='训练') # 使用 tqdm 显示进度条

    for batch in progress_bar:
        # 数据移动到指定设备 (CPU/GPU)
        texts = batch['texts'].to(device)
        labels = batch['labels'].to(device)
        lengths = batch['lengths'].to(device) # lengths 实际未使用，但保持接口一致

        optimizer.zero_grad() # 清除上一轮的梯度

        # 前向传播：获取模型输出 (logits)
        outputs = model(texts, lengths)

        # 计算损失
        loss = criterion(outputs, labels)

        # 反向传播：计算梯度
        loss.backward()

        # 更新模型参数
        optimizer.step()

        total_loss += loss.item() # 累加损失值 (.item() 获取标量值)
        # 更新进度条后缀，显示当前批次的损失
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    # 返回平均训练损失
    return total_loss / len(dataloader)
```

*   设置模型为训练模式 (`model.train()`)。这会启用 Dropout 和 Batch Normalization（如果模型中有的话）的训练行为。
*   初始化 `total_loss`。
*   使用 `tqdm` 包装 `dataloader` 以显示训练进度。
*   **遍历批次**:
    *   从 `dataloader` 获取一个批次的数据。
    *   将数据（文本、标签）移动到 `device` (GPU 或 CPU)。
    *   **梯度清零**: `optimizer.zero_grad()` 清除之前计算的梯度，防止梯度累积。
    *   **前向传播**: `outputs = model(texts, lengths)` 将输入数据传入模型，得到预测结果 (logits)。
    *   **计算损失**: `loss = criterion(outputs, labels)` 使用定义的损失函数（如交叉熵损失 `nn.CrossEntropyLoss`）计算预测结果 `outputs` 和真实标签 `labels` 之间的损失。
    *   **反向传播**: `loss.backward()` 根据损失值自动计算模型中所有可训练参数的梯度。
    *   **参数更新**: `optimizer.step()` 使用优化器（如 Adam）根据计算出的梯度更新模型的权重。
    *   累加当前批次的损失值。
    *   更新 `tqdm` 进度条，显示当前批次的损失。
*   循环结束后，返回整个 epoch 的平均训练损失。

---

### 8. 评估函数 (Evaluation Function)

```python
def evaluate(model, dataloader, criterion, device):
    model.eval() # 设置模型为评估模式（禁用 Dropout 等）
    total_loss = 0
    all_preds = [] # 存储所有预测结果
    all_labels = [] # 存储所有真实标签

    progress_bar = tqdm(dataloader, desc='评估')

    with torch.no_grad(): # 禁用梯度计算，节省内存和计算资源
        for batch in progress_bar:
            # 数据移动到设备
            texts = batch['texts'].to(device)
            labels = batch['labels'].to(device)
            lengths = batch['lengths'].to(device)

            # 前向传播
            outputs = model(texts, lengths)
            # 计算损失
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

            # 获取预测类别 (取 logits 中概率最大的索引)
            _, preds = torch.max(outputs, 1)
            # 将预测结果和真实标签添加到列表中 (移回 CPU 并转为 NumPy 数组)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算准确率
    accuracy = accuracy_score(all_labels, all_preds)
    # 返回平均验证损失、准确率、所有预测标签和所有真实标签
    return total_loss / len(dataloader), accuracy, all_preds, all_labels
```

*   设置模型为评估模式 (`model.eval()`)。这会禁用 Dropout，并让 Batch Normalization 使用运行时的统计数据。
*   初始化 `total_loss` 以及用于存储所有预测和标签的列表 `all_preds`, `all_labels`。
*   使用 `tqdm` 包装 `dataloader`。
*   **`with torch.no_grad():`**: 在这个块内部，PyTorch 不会计算梯度。这在评估阶段是必要的，因为我们不需要更新模型，这样做可以减少内存消耗并加速计算。
*   **遍历批次**:
    *   获取数据并移动到 `device`。
    *   **前向传播**: 获取模型输出 `outputs`。
    *   **计算损失**: 计算当前批次的损失。
    *   累加损失值。
    *   **获取预测**: `_, preds = torch.max(outputs, 1)` 找到 `outputs` (logits) 中每个样本得分最高的那个类别的索引，作为模型的预测结果 `preds`。
    *   将当前批次的预测 `preds` 和真实标签 `labels` 移回 CPU (`.cpu()`)，转换为 NumPy 数组 (`.numpy()`)，并添加到 `all_preds` 和 `all_labels` 列表中。
*   循环结束后，使用 `sklearn.metrics.accuracy_score` 计算整个验证集上的准确率。
*   返回平均验证损失、准确率以及所有预测和真实标签的列表。

---

### 9. 可视化函数 (Plotting Functions)

```python
def plot_confusion_matrix(true_labels, pred_labels):
    # ... (代码省略)
    plt.savefig('confusion_matrix.png')
    plt.show()

def plot_training_history(train_losses, val_losses, accuracies):
    # ... (代码省略)
    plt.savefig('training_history.png')
    plt.show()
```

*   **`plot_confusion_matrix(true_labels, pred_labels)`**:
    *   使用 `sklearn.metrics.confusion_matrix` 计算混淆矩阵。
    *   使用 `seaborn.heatmap` 将混淆矩阵可视化为热力图。`annot=True` 在格子上显示数值，`fmt='d'` 表示整数格式，`cmap='Blues'` 设置颜色主题。
    *   设置 x 轴、y 轴标签和标题。
    *   将图像保存为 `confusion_matrix.png`。
    *   `plt.show()` 显示图像。
*   **`plot_training_history(train_losses, val_losses, accuracies)`**:
    *   创建两个子图。
    *   **子图 1**: 绘制训练损失 (`train_losses`) 和验证损失 (`val_losses`) 随 epoch 变化的曲线。添加图例、坐标轴标签和标题。
    *   **子图 2**: 绘制验证准确率 (`accuracies`) 随 epoch 变化的曲线。添加坐标轴标签和标题。
    *   `plt.tight_layout()` 调整子图布局，防止重叠。
    *   将图像保存为 `training_history.png`。
    *   `plt.show()` 显示图像。

---

### 10. 主函数 (Main Function)

```python
def main():
    print("加载数据集...")
    try:
        # 使用自定义函数加载数据
        train_dataset, test_dataset, vocab = load_imdb_data_manually()
    except Exception as e:
        print(f"无法加载IMDB数据集: {e}")
        return

    # 创建数据集子集 (为了快速演示)
    train_subset_size = min(10000, len(train_dataset))
    test_subset_size = min(2000, len(test_dataset))
    train_subset = torch.utils.data.Subset(train_dataset, range(train_subset_size))
    test_subset = torch.utils.data.Subset(test_dataset, range(test_subset_size))
    print(...) # 打印数据集大小信息

    # 创建数据加载器
    train_dataloader = DataLoader(...)
    test_dataloader = DataLoader(...)

    # 定义模型超参数
    vocab_size = len(vocab)
    embed_dim = 256
    num_heads = 8
    num_layers = 4
    ff_dim = 512
    num_classes = 2
    dropout = 0.1

    # 初始化模型并移动到设备
    model = TransformerClassifier(...) .to(device)

    # 设置优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    # 训练循环
    epochs = 5
    train_losses = []
    val_losses = []
    accuracies = []

    for epoch in range(epochs):
        start_time = time.time()

        # 调用训练和评估函数
        train_loss = train(model, train_dataloader, optimizer, criterion, device)
        val_loss, accuracy, _, _ = evaluate(model, test_dataloader, criterion, device)

        # 记录历史数据
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        accuracies.append(accuracy)

        elapsed_time = time.time() - start_time
        # 打印当前 epoch 的结果
        print(...)

    # 最终评估
    _, final_accuracy, pred_labels, true_labels = evaluate(...)
    print(f"最终测试准确率: {final_accuracy:.4f}")

    # 可视化结果
    print("正在生成混淆矩阵...")
    plot_confusion_matrix(true_labels, pred_labels)
    print("正在生成训练历史图...")
    plot_training_history(train_losses, val_losses, accuracies)

    # 保存模型状态字典
    torch.save(model.state_dict(), "transformer_classifier.pth")
    print("模型已保存到 transformer_classifier.pth")
```

*   **数据加载**: 调用 `load_imdb_data_manually()` 加载数据。包含 `try...except` 块以处理可能的加载错误。
*   **创建子集**: 为了加快训练速度（尤其是在 CPU 上），使用 `torch.utils.data.Subset` 从完整数据集中选取一部分样本进行训练和测试。
*   **创建 DataLoader**: 使用子集创建训练和测试数据加载器。
*   **定义超参数**: 设置模型结构相关的参数，如词汇表大小、嵌入维度、注意力头数、Transformer 层数、前馈网络维度、类别数和 Dropout 率。
*   **初始化模型**: 创建 `TransformerClassifier` 实例，并将模型移动到之前确定的 `device` (CPU 或 GPU)。
*   **设置优化器和损失函数**:
    *   `optimizer`: 选择 Adam 优化器，传入模型的可训练参数 (`model.parameters()`) 和学习率 (`lr`)。
    *   `criterion`: 选择交叉熵损失函数 (`nn.CrossEntropyLoss`)，适用于多分类任务（这里是二分类）。
*   **训练循环**:
    *   设置训练的总轮数 `epochs`。
    *   初始化列表用于存储每轮的训练损失、验证损失和验证准确率。
    *   **按轮次训练**:
        *   记录开始时间。
        *   调用 `train()` 函数执行一轮训练，获取训练损失。
        *   调用 `evaluate()` 函数在测试集上进行评估，获取验证损失和准确率。
        *   将损失和准确率记录到列表中。
        *   计算并打印当前轮次耗时及各项指标。
*   **最终评估**: 训练结束后，再次调用 `evaluate()` 获取最终的测试准确率以及所有预测和真实标签，用于后续可视化。
*   **可视化**: 调用 `plot_confusion_matrix` 和 `plot_training_history` 生成并显示/保存结果图表。
*   **保存模型**: 使用 `torch.save(model.state_dict(), "...")` 将训练好的模型权重（状态字典）保存到文件 `transformer_classifier.pth`，以便将来加载和使用。

---

### 11. 执行入口 (Execution Guard)

```python
if __name__ == "__main__":
    main()
```

*   这是 Python 脚本的标准入口点。
*   `__name__` 是 Python 的一个内置变量。当脚本被直接运行时，`__name__` 的值是 `"__main__"`。如果脚本作为模块被其他脚本导入，则 `__name__` 的值是模块名（文件名）。
*   这个 `if` 语句确保 `main()` 函数只有在脚本被直接执行时才会被调用，而不是在被导入时调用。

