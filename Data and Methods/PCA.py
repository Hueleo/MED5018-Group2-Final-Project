#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# In[2]:


# 从CSV文件中读取数据
file_path = 'cause_of_deaths.csv'
death = pd.read_csv(file_path)

# 显示数据的前几行
print(death.head())


# In[3]:


# 去除非数值变量
data = death.iloc[:, 3:]

# 1. 标准化数据
standardized_data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

# 2. 计算协方差矩阵
covariance_matrix = np.cov(standardized_data, rowvar=False)

# 3. 计算特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

# 4. 选择前两个主成分
num_components = 2
top_eigenvectors = eigenvectors[:, :num_components]

# 5. 构建投影矩阵
projection_matrix = top_eigenvectors

# 6. 投影
pca_result = np.dot(standardized_data, projection_matrix)
pca_results = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
# pca_result 中的每一行包含了每个国家在主成分上的得分,并保存到文件中
print(pca_results)
pca_results.to_csv('pca_results.csv', index=False)


# In[4]:


# 绘制散点图
plt.scatter(pca_result[:, 0], pca_result[:, 1])

# 添加标题和轴标签
plt.title('Scatter Plot of PCA Results')
plt.xlabel('Principal Component 1 (PC1)')
plt.ylabel('Principal Component 2 (PC2)')

# 显示图形
plt.show()


# In[5]:


# 绘制带有聚类结果的散点图
# 使用K均值聚类算法
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(pca_result)

# 将聚类结果添加到数据中
data_with_clusters = data.copy()
data_with_clusters['Cluster'] = clusters

sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=clusters, palette='viridis')

# 添加标题和轴标签
plt.title('Scatter Plot with Clusters After PCA')
plt.xlabel('Principal Component 1 (PC1)')
plt.ylabel('Principal Component 2 (PC2)')

# 显示图形
plt.show()


# In[ ]:




