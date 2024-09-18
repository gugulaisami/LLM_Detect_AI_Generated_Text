import os
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# 数据文件路径
#train_essays_path = 'archive/train_v2_drcat_02.csv'
train_essays_path = 'llm-detect-ai-generated-text/train_essays.csv'
test_essays_path = 'llm-detect-ai-generated-text/test_essays.csv'
train_prompts_path = 'llm-detect-ai-generated-text/train_prompts.csv'

# 读取数据
test = pd.read_csv('llm-detect-ai-generated-text/test_essays.csv')
train_essays_df = pd.read_csv(train_essays_path)
test_essays_df = pd.read_csv(test_essays_path)
train_prompts_df = pd.read_csv(train_prompts_path)

# 文本预处理函数
def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    text = re.sub(r'\d+', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

# 预处理文章
train_essays_df['preprocessed_text'] = train_essays_df['text'].apply(preprocess_text)
test_essays_df['preprocessed_text'] = test_essays_df['text'].apply(preprocess_text)

# 文章向量化
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
tfidf_matrix = tfidf_vectorizer.fit_transform(train_essays_df['preprocessed_text'])
test_tfidf_matrix = tfidf_vectorizer.transform(test_essays_df['preprocessed_text'])

# 利用轮廓系数确定最佳簇的数量
range_n_clusters = list(range(2, 10))
silhouette_avg = []
for num_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(tfidf_matrix)
    cluster_labels = kmeans.labels_
    silhouette_avg.append(silhouette_score(tfidf_matrix, cluster_labels))

# 识别具有最高轮廓系数的簇的数量
optimal_num_clusters = np.argmax(silhouette_avg) + 2  # 加2是因为范围从2开始
print(f"最佳簇数量: {optimal_num_clusters}")

# 绘制轮廓系数
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(10, 6))
plt.plot(range_n_clusters, silhouette_avg, 'bx-')
plt.xlabel('簇的数量')
plt.ylabel('轮廓系数')
plt.title('不同簇数量的轮廓系数')
plt.show()

# 应用KMeans聚类
kmeans = KMeans(n_clusters=optimal_num_clusters, random_state=1)
clusters = kmeans.fit_predict(tfidf_matrix)

# 利用PCA将数据降维到2维以便可视化
pca = PCA(n_components=2)
reduced_X_sample = pca.fit_transform(tfidf_matrix.toarray())

# 创建用于绘图的数据框
pca_df = pd.DataFrame(reduced_X_sample, columns=['PC1', 'PC2'])
#pca_df['generated'] = train_essays_df['generated']
pca_df['generated'] = train_essays_df['generated'].values  # 使用 .values 来避免索引不匹配
# 自定义调色板用于绘图
custom_palette = {0: "red", 1: "blue"}
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 绘制PCA结果
plt.figure(figsize=(14, 7))
sns.scatterplot(
    x='PC1', y='PC2', hue='generated', data=pca_df,
    palette=custom_palette, alpha=0.7
)
plt.title('文章的PCA (2个主成分)')
plt.xlabel('主成分 1')
plt.ylabel('主成分 2')
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles, ['学生', 'LLM 生成'], title='文章类型')
plt.show()


# 分割数据集并应用KMeans聚类
X_train, X_test, y_train, y_test = train_test_split(reduced_X_sample, train_essays_df['generated'], test_size=0.2, random_state=42)

# 在训练集上应用KMeans聚类
kmeans = KMeans(n_clusters=optimal_num_clusters, random_state=42)
kmeans.fit(X_train)
train_clusters = kmeans.predict(X_train)
test_clusters = kmeans.predict(X_test)

# 在绘制PCA结果之前，将索引重置
pca_df = pca_df.reset_index(drop=True)

# 绘制带有KMeans聚类的测试集PCA结果
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(14, 7))
sns.scatterplot(
    x=pca_df['PC1'], y=pca_df['PC2'], hue=test_clusters,
    palette=custom_palette, alpha=0.7
)
plt.title('带有KMeans聚类的测试集文章的PCA')
plt.xlabel('主成分 1')
plt.ylabel('主成分 2')
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles, [f'簇 {i}' for i in range(optimal_num_clusters)], title='簇')
plt.show()

# 在测试集上应用PCA
reduced_test_X = pca.transform(test_tfidf_matrix.toarray())

# 计算训练集上的准确率
train_accuracy = accuracy_score(y_train, train_clusters)
print(f"训练集准确率: {train_accuracy:.2%}")

# 计算测试集上的准确率
test_accuracy = accuracy_score(y_test, test_clusters)
print(f"验证集准确率: {test_accuracy:.2%}")

# 在整个数据集上应用KMeans聚类进行提交
kmeans = KMeans(n_clusters=optimal_num_clusters, random_state=42)
kmeans.fit(reduced_X_sample)

# 预测测试集的簇
preds_test = kmeans.predict(reduced_test_X)

# 将预测的簇添加到测试数据框中
test['generated'] = preds_test

# 创建提交文件
submission = pd.DataFrame({
    'id': test["id"],
    'generated': test['generated']
})

# 保存提交文件
submission.to_csv('submission.csv', index=False)
