import sys
import gc
import traceback
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)
from datasets import Dataset
from transformers import PreTrainedTokenizerFast
from collections import Counter
import itertools

# 读取数据
test = pd.read_csv('llm-detect-ai-generated-text/test_essays.csv')
#train = pd.read_csv("archive/train_v2_drcat_02.csv", sep=',')
train = pd.read_csv('llm-detect-ai-generated-text/train_essays.csv')

#打印数据
print(test.tail())
print(train.tail())

#查找空缺数值
missing_values_train = train.isnull().sum()
print("missing_values_train:", missing_values_train)
missing_values_prompts = test.isnull().sum()
print("\nmissing_values_test:", missing_values_prompts)

# 删除重复的训练数据
train = train.drop_duplicates(subset=['text']).reset_index(drop=True)

# 绘制训练集标签分布图
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(8, 6))
#sns.countplot(x='label', data=train)
sns.countplot(x='generated', data=train)
plt.title('训练集标签分布')
plt.xlabel('标签')
plt.ylabel('频率')
plt.show()

# 绘制论文长度分布图
train['essay_length'] = train['text'].apply(len)
sns.set(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(14, 7))
# sns.histplot(train[train['label'] == 0]['essay_length'], color="skyblue", label='学生作文', kde=True)
# sns.histplot(train[train['label'] == 1]['essay_length'], color="red", label='LLM生成作文', kde=True)
sns.histplot(train[train['generated'] == 0]['essay_length'], color="skyblue", label='学生作文', kde=True)
sns.histplot(train[train['generated'] == 1]['essay_length'], color="red", label='LLM生成作文', kde=True)
plt.title('论文长度分布')
plt.xlabel('论文长度（字符数）')
plt.ylabel('频率')
plt.legend()
plt.show()

# 绘制分类比较论文长度图
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(12, 8))
#sns.boxplot(x='label', y='essay_length', data=train)
sns.boxplot(x='generated', y='essay_length', data=train)
plt.title('论文来源的长度比较')
plt.xlabel('论文来源')
plt.ylabel('论文长度')
plt.xticks([0, 1], ['学生写作', 'LLM生成'])
plt.show()

# 绘制学生和AI的最常用词汇统计图
def plot_most_common_words(text_series, num_words=30, title="最常用词汇"):
    all_text = ' '.join(text_series).lower()
    words = all_text.split()
    word_freq = Counter(words)
    common_words = word_freq.most_common(num_words)

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(15, 6))
    sns.barplot(x=[word for word, freq in common_words], y=[freq for word, freq in common_words])
    plt.title(title)
    plt.xticks(rotation=45)
    plt.xlabel('词汇')
    plt.ylabel('频率')
    plt.show()

plot_most_common_words(train[train['generated'] == 0]['text'],
                       title="学生作文中最常用词汇")
plot_most_common_words(train[train['generated'] == 1]['text'],
                        title="LLM生成作文中最常用词汇")
#                        plot_most_common_words(train[train['label'] == 0]['text'],
#                        title="学生作文中最常用词汇")
# plot_most_common_words(train[train['label'] == 1]['text'],
#                        title="LLM生成作文中最常用词汇")
# 定义虚拟函数
def dummy(text):
    return text

# 创建 Byte-Pair Encoding tokenizer
raw_tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
raw_tokenizer.normalizer = normalizers.Sequence([normalizers.NFC()] + [normalizers.Lowercase()])
raw_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
trainer = trainers.BpeTrainer(vocab_size=30522, special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"])
dataset = Dataset.from_pandas(train[['text']])
def train_corp_iter():
    for i in range(0, len(dataset), 1000):
        yield dataset[i : i + 1000]["text"]
raw_tokenizer.train_from_iterator(train_corp_iter(), trainer=trainer)
tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=raw_tokenizer,
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]",
)

# 分词化训练集文本
tokenized_texts_train = []
for text in train['text'].tolist():
    tokens = tokenizer.tokenize(text)
    tokenized_texts_train.append(tokens)

tokenized_texts_test = []
for text in test['text'].tolist():
    tokens = tokenizer.tokenize(text)
    tokenized_texts_test.append(tokens)

# 合并训练和测试数据
df = pd.concat([train, test], axis=0)

# 使用TF-IDF向量化器提取文本特征
vectorizer = TfidfVectorizer(ngram_range=(3, 5), lowercase=False, sublinear_tf=True,
                             analyzer='word', tokenizer=dummy, preprocessor=dummy,
                             token_pattern=None, strip_accents='unicode',
                             min_df=1, max_df=0.95)
X = vectorizer.fit_transform(tokenized_texts_train)

# 使用相同的向量化器提取测试集文本特征
tf_test = vectorizer.transform(tokenized_texts_test)

# 划分训练集和测试集
#X_train, X_test, y_train, y_test = train_test_split(X[:train.shape[0]], train['label'], test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X[:train.shape[0]], train['generated'], test_size=0.2, random_state=42)

# print(X_train[:5])  # 查看前5行
# print(50*'-')
# print(tf_test[:5])  # 查看前5行
# print(50*'-')

# 初始化多个分类器
lr = LogisticRegression()
mnb = MultinomialNB(alpha=0.02)
svm = SVC(probability=True)

# 初始化投票分类器
ensemble = VotingClassifier(estimators=[('lr', lr), ('mnb', mnb), ('svm', svm)], voting='soft')

# 交叉验证评估模型性能
scorer = make_scorer(accuracy_score)
cv_scores = cross_val_score(ensemble, X_train, y_train, cv=5, scoring=scorer)
print("Cross-Validation Mean Accuracy:", np.mean(cv_scores))

# 在整个训练集上拟合模型
ensemble.fit(X_train, y_train)

# 在验证集上进行预测
y_pred_test = ensemble.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred_test)
print("validaton Set Accuracy:", test_accuracy)

# 在训练集上进行预测
y_pred_train = ensemble.predict(X_train)
train_accuracy = accuracy_score(y_train, y_pred_train)
print("Train Set Accuracy:", train_accuracy)

#使用训练好的模型对测试数据进行预测
preds_test = ensemble.predict_proba(tf_test)[:, 1]

#添加生成文本的预测概率到测试数据中
test['generated'] = preds_test

# 创建提交文件
submission = pd.DataFrame({
    'id': test["id"],
    'generated': test['generated']
})

# 保存提交文件
submission.to_csv('submission.csv', index=False)

