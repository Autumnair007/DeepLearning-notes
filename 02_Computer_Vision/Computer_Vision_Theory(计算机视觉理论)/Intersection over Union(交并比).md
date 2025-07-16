#  IoU笔记（Gemini2.5Pro生成）

学习资料：[语义分割的评价指标——IoU_语义分割iou-CSDN博客](https://blog.csdn.net/lingzhou33/article/details/87901365)

------

在图像分割领域，**IoU** 是一个非常重要的概念，它的全称是 **Intersection over Union**，中文意思是“**交并比**”。它是一种用来评估图像分割模型性能的标准度量指标 [[1]](https://blog.csdn.net/lingzhou33/article/details/87901365)。

### IoU 是如何计算的？

IoU 的计算方式是：模型预测的分割区域（Prediction）与真实的物体的区域（Ground Truth）之间的**交集**面积，再除以它们之间的**并集**面积。

**公式：** `IoU = Area of Intersection / Area of Union`

*   **交集 (Intersection)**：模型预测正确的区域。
*   **并集 (Union)**：模型预测的区域加上真实区域的总和（重叠部分只计算一次）。

### IoU 代表什么？

IoU 的值域在 0 到 1 之间：
*   **IoU = 1**：表示模型的预测结果与真实区域完美重合，是理想的最高值。
*   **IoU = 0**：表示预测区域与真实区域没有任何重叠。

简单来说，**IoU 的值越高，说明模型的分割结果越精确** [[2]](https://zhuanlan.zhihu.com/p/378796770)。

### 为什么它很重要？

在图像分割、目标检测等计算机视觉任务中，IoU 被广泛用于：
1.  **评估模型精度**：它是衡量模型预测的边界位置有多准确的关键指标 [[3]](https://developer.baidu.com/article/detail.html?id=1892805)。
2.  **设定评判标准**：在很多竞赛和论文中，通常会设定一个 IoU 阈值（比如 0.5），只有当模型的预测结果 IoU 大于这个阈值时，才被认为是一次成功的检测或分割。

总之，IoU 是一个直观且有效的指标，用于量化模型在定位和描绘图像中特定对象轮廓方面的准确性 [[4]](https://www.oryoy.com/news/jie-mi-iou-shen-du-jie-xi-fen-ge-suan-fa-zhong-de-jing-sui-yu-shi-zhan-ji-qiao.html)。
