# ========================
# 文本预处理完整流程（带本地文件缓存）
# 功能：下载文本→保存到本地→加载→清洗→分词→构建词表→编码
# ========================

import os  # 导入os模块，用于文件路径操作和检查
import requests  # 导入requests模块，用于发送HTTP请求
import re  # 导入正则表达式模块，用于文本清洗
from collections import Counter  # 导入Counter类，用于词频统计

# ========================
# 1. 数据下载模块
# ========================
def download_text(url, save_path):
    """下载文本并保存到指定路径
    Args:
        url: 下载链接
        save_path: 本地保存路径（如 'alice_original.txt'）
    Returns:
        str: 文本内容
    """
    # 如果文件已存在，直接读取文件内容
    if os.path.exists(save_path):
        with open(save_path, 'r', encoding='utf-8') as f:
            print(f"从本地加载缓存文件: {save_path}")  # 输出提示信息
            return f.read()  # 返回文件内容

    # 文件不存在则发送HTTP请求下载
    response = requests.get(url)  # 发送GET请求获取网页内容
    response.raise_for_status()  # 检查请求是否成功，如果失败则抛出异常
    text = response.text  # 获取响应文本

    # 将下载的内容保存到本地文件
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(text)  # 写入文件
    print(f"文件已下载并保存到: {os.path.abspath(save_path)}")  # 输出绝对路径
    return text  # 返回文本内容

# 文件路径配置
alice_url = "https://www.gutenberg.org/files/11/11-0.txt"  # Alice in Wonderland的URL
original_file = "alice_original.txt"  # 原始文本保存路径
processed_file = "alice_processed.txt"  # 清洗后文本保存路径

# 下载/加载原始文本
raw_text = download_text(alice_url, original_file)  # 调用下载函数获取文本

# ========================
# 2. 数据预处理模块
# ========================
# 提取正文（去除文档头尾的版权信息等）
start_marker = "CHAPTER I"  # 正文开始标记
end_marker = "End of Project Gutenberg"  # 正文结束标记
text = raw_text[raw_text.find(start_marker):raw_text.find(end_marker)]  # 切片提取正文内容

def clean_and_save(text, save_path):
    """清洗文本并保存到文件
    Args:
        text: 原始文本
        save_path: 清洗后文本保存路径
    Returns:
        str: 清洗后的文本
    """
    # 清洗文本：转小写、只保留英文字母和部分标点符号
    cleaned = re.sub(r'[^a-zA-Z\s.,!?]', '', text.lower())  # 去除非字母和常用标点
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()  # 合并多个空格为单个空格并去除首尾空格

    # 保存清洗后的文本到文件
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(cleaned)  # 写入文件
    print(f"清洗后文本已保存到: {os.path.abspath(save_path)}")  # 输出保存位置
    return cleaned  # 返回清洗后的文本

# 执行清洗并保存
cleaned_text = clean_and_save(text, processed_file)  # 清洗文本并保存
print("\n清洗后片段:", cleaned_text[:200] + "...")  # 打印清洗后文本的前200个字符

def load_processed_text(file_path):
    """从本地加载已处理的文本
    Args:
        file_path: 文件路径
    Returns:
        str: 文件内容
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()  # 读取并返回文件内容

# 从本地加载处理后的文本（演示如何重新加载）
reloaded_text = load_processed_text(processed_file)  # 加载已处理的文本
print("\n重新加载文本长度:", len(reloaded_text))  # 打印文本长度

# ========================
# 3. 词元化模块（分词）
# ========================
def tokenize(text, mode='word'):
    """分词函数
    Args:
        text: 清洗后的文本
        mode: 分词模式 ('word'或'char')
    Returns:
        list: 词元列表
    """
    if mode == 'word':
        # 简单空格分词（英文适用）
        return text.split()  # 使用空格分割文本
    elif mode == 'char':
        # 字符级分词
        return list(text)  # 将文本转换为字符列表
    else:
        raise ValueError("模式需为 'word' 或 'char'")  # 抛出无效模式的错误

# 单词级分词示例
word_tokens = tokenize(cleaned_text, mode='word')  # 进行单词级分词
print("\n单词词元示例:", word_tokens[:10])  # 打印前10个单词

# ========================
# 4. 词表构建模块
# ========================
class Vocabulary:
    """词表类
    功能：
    1. 统计词频
    2. 建立词元到索引的映射
    3. 处理未知词元
    """

    def __init__(self, tokens, min_freq=1, unk_token="<unk>"):
        """初始化词表
        Args:
            tokens: 所有词元的列表
            min_freq: 最小词频阈值（低于此频率的词将被忽略）
            unk_token: 未知词标记
        """
        # 使用Counter统计词频（返回{词元:频率}的字典）
        counter = Counter(tokens)  # 统计每个词元的出现次数
        # 按频率降序排序（元素为(词元,频率)元组的列表）
        self.token_freq = counter.most_common()  # 按词频降序排列

        # 初始化词表：首元素为未知词标记
        self.idx_to_token = [unk_token]  # 索引到词元的映射列表，0索引为未知词
        # 添加满足最小词频的词元
        self.idx_to_token += [token for token, freq in self.token_freq if freq >= min_freq]  # 过滤低频词

        # 创建词元到索引的映射字典
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}  # 词元到索引的映射
        self.unk_token = unk_token  # 保存未知词标记

    def __len__(self):
        """返回词表大小"""
        return len(self.idx_to_token)  # 返回词表中的词元数量

    def encode(self, tokens):
        """将词元序列转换为数字索引序列
        Args:
            tokens: 词元列表
        Returns:
            list: 索引列表
        """
        return [self.token_to_idx.get(token, 0) for token in tokens]  # 若词元不在词表中，返回未知词索引0

    def decode(self, indices):
        """将数字索引序列转换回词元序列
        Args:
            indices: 索引列表
        Returns:
            list: 词元列表
        """
        return [self.idx_to_token[idx] for idx in indices]  # 将每个索引转换为对应的词元


# 构建词表示例（过滤词频<3的低频词）
word_vocab = Vocabulary(word_tokens, min_freq=3)  # 创建词表实例，最小词频为3
print("\n词表大小:", len(word_vocab))  # 打印词表大小
print("高频词示例:", word_vocab.idx_to_token[:10])  # 打印前10个高频词

# ========================
# 5. 编码转换模块
# ========================
# 将前15个词元转换为索引
encoded_sequence = word_vocab.encode(word_tokens[:15])  # 编码前15个词元
print("\n编码示例:", encoded_sequence)  # 打印编码结果
# 将索引转换回文本（演示可逆性）
print("解码回文本:", " ".join(word_vocab.decode(encoded_sequence)))  # 解码并打印结果


# ========================
# 6. 完整流程整合
# ========================
def process_text_pipeline(url, original_save_path, processed_save_path, mode='word', min_freq=2):
    """端到端文本处理管道
    Args:
        url: 文本下载链接
        original_save_path: 原始文本保存路径
        processed_save_path: 处理后文本保存路径
        mode: 分词模式
        min_freq: 最小词频
    Returns:
        tuple: (数字索引列表, 词表对象)
    """
    # 数据下载
    raw_text = download_text(url, original_save_path)  # 下载并保存原始文本
    
    # 提取正文
    start_marker = "CHAPTER I"  # 正文开始标记
    end_marker = "End of Project Gutenberg"  # 正文结束标记
    extract_text = raw_text[raw_text.find(start_marker):raw_text.find(end_marker)]  # 提取正文
    
    # 清洗文本
    cleaned = clean_and_save(extract_text, processed_save_path)  # 清洗并保存文本
    
    # 分词
    tokens = tokenize(cleaned, mode=mode)  # 将文本分词为词元列表

    # 构建词表
    vocab = Vocabulary(tokens, min_freq=min_freq)  # 创建词表对象
    
    # 编码转换
    indices = vocab.encode(tokens)  # 将词元编码为索引

    return indices, vocab  # 返回索引列表和词表对象

# 执行完整流程
final_indices, final_vocab = process_text_pipeline(
    alice_url, 
    original_file, 
    processed_file
)  # 运行完整处理流程
print("\n最终序列长度:", len(final_indices))  # 打印最终序列长度
print("前20个索引:", final_indices[:20])  # 打印前20个编码索引

# 如果作为主程序运行，则执行以下代码
if __name__ == "__main__":
    print("\n文本预处理完成！")  # 打印完成消息
