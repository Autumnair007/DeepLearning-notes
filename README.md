# DeepLearning-notes
 Personal study notes repository about deep learning.

------

记录一下自己学的模型，让AI (Gemini2.5 Pro 和 Claude 3.7 Sonnet Thinking) 跑一下相关代码和相关的知识点整理一下。笔记内容可能存在错误，需要经常看看然后纠正。

------

参考资料：[前言 — 动手学深度学习 2.0.0 documentation](https://zh-v2.d2l.ai/chapter_preface/index.html)

[贺完结！CS231n官方笔记授权翻译总集篇发布 - 知乎](https://zhuanlan.zhihu.com/p/21930884)

------

环境配置命令：

conda create -n dl python=3.8

conda activate dl

conda install pytorch\==1.12.1 torchvision\==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge

pip install pandas matplotlib scikit-learn seaborn tqdm nltk gensim

(注意这里不能用conda，如果使用conda安装会导致numpy包冲突而无法运行某些代码。如果网络报错有可能是梯子的问题，需要注意。)

