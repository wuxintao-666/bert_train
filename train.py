import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONWARNINGS"] = "ignore"

import warnings
warnings.filterwarnings("ignore")

# 设置随机种子
seed = 42

# 设置Python随机种子
import random
random.seed(seed)

# 设置NumPy随机种子
import numpy as np
np.random.seed(seed)

# 设置PyTorch随机种子
import torch
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 设置transformers日志级别
from transformers.utils import logging
logging.set_verbosity_error()
import torch.nn as nn
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
# 1. 读取数据
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("开始读数据...")
df = pd.read_csv('devgptemotion.csv')  # 替换为你的文件名
rate = [1, 2, 4]  # negative:positive:neutral
def resample_fixed_ratio(df, target_ratios, random_state=42):
    target_order = ['negative', 'positive', 'neutral']  # 指定类别顺序
    
    if len(target_order) != len(target_ratios):
        raise ValueError("类别数和目标比例长度不一致")

    class_counts = df['PromptEmotion'].value_counts()

    # 计算最大可行倍数 k
    k_values = []
    for cls, ratio in zip(target_order, target_ratios):
        k_values.append(class_counts[cls] / ratio)
    k = min(k_values)

    k = np.floor(k)

    sampled_dfs = []
    for cls, ratio in zip(target_order, target_ratios):
        target_size = int(np.floor(k * ratio))   # 向下取整
        cls_df = df[df['PromptEmotion'] == cls]
        sampled = cls_df.sample(target_size, random_state=random_state, replace=False)
        sampled_dfs.append(sampled)

    return pd.concat(sampled_dfs).sample(frac=1, random_state=random_state)

df = resample_fixed_ratio(df, rate)
# 不采样
df_0 = df[df['PromptEmotion'] == 'positive']
df_1 = df[df['PromptEmotion'] == 'negative']
df_2 = df[df['PromptEmotion'] == 'neutral']
df_small = pd.concat([df_0, df_1, df_2]).reset_index(drop=True)
print(f'采样后数据集总数: {len(df_small)}')
print(df_small['PromptEmotion'].value_counts())
df = df_small

# 2. 划分训练集、验证集和测试集
# 首先划分训练+验证集(80%)和测试集(20%)
train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['PromptEmotion'])

# 再将训练+验证集划分为训练集(80%)和验证集(20%)
train_df, val_df = train_test_split(train_val_df, test_size=0.25, random_state=42, stratify=train_val_df['PromptEmotion'])  # 0.25*0.8=0.2

# 将标签转换为数值
label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
train_df['label'] = train_df['PromptEmotion'].map(label_map)
val_df['label'] = val_df['PromptEmotion'].map(label_map)
test_df['label'] = test_df['PromptEmotion'].map(label_map)

# 计算类别权重
labels = train_df['label'].values
class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights = torch.FloatTensor(class_weights).to(device)
print("类别权重：", class_weights)

# 打印数据集分布
print("\n===== 数据集分布 =====")
print(f"原始数据集总数: {len(df)}")
print(f"训练集大小: {len(train_df)}")
print(f"验证集大小: {len(val_df)}")
print(f"测试集大小: {len(test_df)}")

print("\n原始数据集各类别数量：")
print(df['PromptEmotion'].value_counts())
print("\n训练集各类别数量：")
print(train_df['PromptEmotion'].value_counts())
print("\n验证集各类别数量：")
print(val_df['PromptEmotion'].value_counts())
print("\n测试集各类别数量：")
print(test_df['PromptEmotion'].value_counts())
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha[targets] * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
# 3. 数据集类
class SentimentDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128):
        self.texts = df['Prompt'].tolist()
        self.labels = df['label'].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def resolve_model_path(preferred_local_dir: str, repo_id: str):
    """优先使用本地目录；若不存在则回退到在线仓库 ID。"""
    if os.path.isdir(preferred_local_dir):
        return preferred_local_dir
    print(f"未发现本地目录 {preferred_local_dir}，将从在线仓库 {repo_id} 加载并保存到本地。")
    return repo_id

# 4. 加载Tokenizer和模型
preferred_local_dir = './bert-base-uncased'
model_id_or_path = resolve_model_path(preferred_local_dir, 'bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained(model_id_or_path)
model = BertForSequenceClassification.from_pretrained(
    model_id_or_path,
    num_labels=3,
    ignore_mismatched_sizes=True
)
model = model.to(device)

# 若是从在线仓库加载，则将权重与分词器保存到本地目录，便于离线复用
if model_id_or_path != preferred_local_dir:
    os.makedirs(preferred_local_dir, exist_ok=True)
    tokenizer.save_pretrained(preferred_local_dir)
    model.save_pretrained(preferred_local_dir)

# 5. DataLoader
train_dataset = SentimentDataset(train_df, tokenizer)
val_dataset = SentimentDataset(val_df, tokenizer)
test_dataset = SentimentDataset(test_df, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

# 6. 优化器和调度器
optimizer = AdamW(model.parameters(), lr=2e-5)
total_steps = len(train_loader) * 40  # 40个epoch
scheduler = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=int(0.1 * total_steps),  # 10%的步数用于预热
    num_training_steps=total_steps
)

patience = 6  # 早停容忍轮数
counter = 0   # 没有提升的轮数
best_f1 = 0   # 记录最佳验证集Macro-F1
best_model_path = './bert_sentiment_best_model'  # 模型保存路径

# 创建目录保存模型
os.makedirs(best_model_path, exist_ok=True)

# 训练历史记录
history = {
    'epoch': [],
    'train_loss': [],
    'val_macro_f1': [],
    'val_f1_negative': [],
    'val_f1_neutral': [],
    'val_f1_positive': []
}

# 训练循环
for epoch in range(50):
    model.train()
    total_loss = 0
    loop = tqdm(train_loader, desc=f'Epoch {epoch+1}')
    for batch in loop:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # 前向传播
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        # 使用类别权重计算损失
        lossfunc = torch.nn.CrossEntropyLoss(weight=class_weights)
        loss = lossfunc(logits, labels)
        total_loss += loss.item()
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪防止爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        # 更新进度条
        loop.set_postfix(loss=loss.item())

    # 每个epoch后在验证集上评估
    model.eval()
    val_preds, val_trues = [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='验证集评估'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 预测
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # 收集预测结果
            val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            val_trues.extend(labels.cpu().numpy())
    
    # 计算验证集评估指标
    val_macro_f1 = f1_score(val_trues, val_preds, average='macro')
    
    # 计算各类别F1分数
    val_f1_scores = f1_score(val_trues, val_preds, average=None)
    
    # 记录历史
    history['epoch'].append(epoch+1)
    history['train_loss'].append(total_loss/len(train_loader))
    history['val_macro_f1'].append(val_macro_f1)
    history['val_f1_negative'].append(val_f1_scores[0])
    history['val_f1_neutral'].append(val_f1_scores[1])
    history['val_f1_positive'].append(val_f1_scores[2])
    
    # 打印结果
    print(f'\nEpoch {epoch+1} =================')
    print(f'训练损失: {total_loss/len(train_loader):.4f}')
    print(f'验证集Macro-F1: {val_macro_f1:.4f}')
    print('验证集详细分类报告:')
    print(classification_report(
        val_trues, val_preds, 
        target_names=['negative', 'neutral', 'positive'],
        digits=4
    ))
    
    # 检查是否是最佳模型
    if val_macro_f1 > best_f1:
        best_f1 = val_macro_f1
        counter = 0
        model.save_pretrained(best_model_path)
        tokenizer.save_pretrained(best_model_path)
        print(f'>>> 新最佳模型 | 验证集Macro-F1: {val_macro_f1:.4f} | '
              f'Negative-F1: {val_f1_scores[0]:.4f} | '
              f'Neutral-F1: {val_f1_scores[1]:.4f} | '
              f'Positive-F1: {val_f1_scores[2]:.4f}')
    else:
        counter += 1
        print(f'>>> 验证集Macro-F1未提升，早停计数: {counter}/{patience}')
        if counter >= patience:
            print('>>> 早停触发，训练提前终止。')
            break

# 最终加载最佳模型在测试集上进行评估
print("\n===== 最终测试集评估 =====")
model = BertForSequenceClassification.from_pretrained(best_model_path)
model = model.to(device)
model.eval()

test_preds, test_trues = [], []
with torch.no_grad():
    for batch in tqdm(test_loader, desc='测试集评估'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        test_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        test_trues.extend(labels.cpu().numpy())

# 计算测试集评估指标
test_acc = accuracy_score(test_trues, test_preds)
test_macro_f1 = f1_score(test_trues, test_preds, average='macro')
test_f1_scores = f1_score(test_trues, test_preds, average=None)

print(f"\n最佳模型在测试集上的表现:")
print(f"准确率: {test_acc:.4f}")
print(f"Macro-F1: {test_macro_f1:.4f}")
print(f"Negative-F1: {test_f1_scores[0]:.4f}")
print(f"Neutral-F1: {test_f1_scores[1]:.4f}")
print(f"Positive-F1: {test_f1_scores[2]:.4f}")
print(classification_report(
    test_trues, test_preds, 
    target_names=['negative', 'neutral', 'positive'],
    digits=4
))

# 计算并绘制混淆矩阵
cm = confusion_matrix(test_trues, test_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['negative', 'neutral', 'positive'],
            yticklabels=['negative', 'neutral', 'positive'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("\n混淆矩阵已保存为 confusion_matrix.png")

# 保存训练历史
history_df = pd.DataFrame(history)
history_df.to_csv('training_history.csv', index=False)
print("训练历史已保存到 training_history.csv")