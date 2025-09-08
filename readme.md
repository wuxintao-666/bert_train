# BERT情感分析项目指南

## 环境配置
```bash
python -m venv bert_env
bert_env\Scripts\activate
pip install -r requirements.txt
```

## 模型下载
```bash
python download_models.py
```

## 训练参数说明(train.py)
- `--rate`: 调节训练集比例（默认1:2:4）
- `--df`: 指定数据集路径（默认'devgptemotion.csv'）
- `--best_model_path`: 不同模型/比例需指定不同保存路径

## 训练执行
```bash
python train.py
```

## 实验结果记录要求
1. 初始数据比例
2. 混淆矩阵图（confusion_matrix.png）
3. 测试集评估结果


⚠️ 注意：每次实验参数变更后请及时更新文档记录