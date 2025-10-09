# 计算机视觉知识枢纽

---

## 🗺️ 核心领域地图 (手动维护)

这里是你手动建立的核心概念导航，就像地铁线路图。

- **理论基础**: [[Digital_Image_Processing/README|数字图像处理]], [[CV中的Transformer核心概念解析]], [[class_imbalance(类别不平衡)]]
- **核心任务**:
    - **图像分类**: [[_Image_Classification_Index]]
    - **语义分割**: [[_Semantic_Segmentation_Index]]
    - **生成模型**: [[_Generative_Models_Index]]
    - **自监督学习**: [[_Self_Supervised_Learning_Index]]

---

## 📚 论文笔记总览 (Dataview 自动生成)

所有计算机视觉领域的论文笔记都会自动出现在这里。

```dataview
TABLE model AS "相关模型", year AS "年份", publication AS "发表会议", tags AS "标签"
FROM "02_Computer_Vision"
WHERE type = "paper-note"
SORT year DESC
```

---

## 💡 概念笔记总览 (Dataview 自动生成)

```dataview
TABLE task AS "相关任务", tags AS "标签"
FROM "02_Computer_Vision"
WHERE type = "concept-note"
```