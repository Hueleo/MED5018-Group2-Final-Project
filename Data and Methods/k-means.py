from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 输出设置
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)

columns = ['Country', 'Code', "Year", 'Meningitis', 'Alzheimers', 'Parkinson',	'Nutritional', 'Malaria', 'Drowning',	'Interpersonal', 'Maternal', 'HIV',	'Drug',
           'Tuberculosis', 'Cardiovascular', 'Lower', 'Neonatal', 'Alcohol', 'Self-harm', 'Exposure', 'Diarrheal', 'Environmental',
           'Neoplasms',	'Conflict',	'Diabetes',	'Chronic_K', 'Poisonings', 'Protein', 'Road', 'Chronic_R', 'Cirrhosis',	'Digestive', 'Fire', 'Acute']

data = pd.read_csv('cause_of_deaths.csv', names=columns, encoding='gbk', low_memory=False, skiprows=[0])  # 48个特征,
# data = pd.read_csv('/tmp/zz/data/K_data_hy.csv', names=column, encoding='gbk', low_memory=False, skiprows=[0])  # 48个特征,
# 删除包含NaN值得行
count = len(data)



print(data)
data = data.dropna()
print('data-shape', data.shape)
# data.set_index('id', inplace=True)
X = data[columns[3:34]]
print(X)

""" K_means """
cost = []
# 定义要测试的聚类数量范围
k_values = range(1, 10)

# 对每个k值运行KMeans算法，并计算惯性得分
for k in k_values:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=0)
    kmeans.fit(X)
    cost.append(kmeans.inertia_)

# 绘制代价与k值的关系图
plt.plot(k_values, cost)
plt.xlabel('Number of clusters')
plt.ylabel('Cost (Inertia)')
plt.title('Elbow method for optimal k')
plt.show()


best_cluster_num = 3
kmeans = KMeans(n_clusters=best_cluster_num, init='k-means++')
# 应用聚类算法
kmeans.fit(X)
# 获取每个样本的标签
labels = kmeans.labels_

# 获取每个群体的中心点
centers = kmeans.cluster_centers_
# 打印聚类结果
for i in range(best_cluster_num):
    print("Cluster ", i)
    print("--------------------")
    print(X[labels == i].describe())
    print("\n")


# 绘制散点图
cmap = plt.cm.get_cmap('viridis', 8)
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, cmap=cmap)
plt.xlabel('Meningitis')
plt.ylabel('Alzheimers')
# 添加颜色条
plt.colorbar()
plt.show()


# 获取每个簇的大小
sizes1 = pd.Series(labels).value_counts()
# 绘制簇大小直方图
sizes1.plot(kind='bar', edgecolor='black')
plt.xlabel('Cluster')
plt.ylabel('Size')
plt.show()

# 创建树状聚类图
Z = linkage(kmeans.cluster_centers_, method='ward')   # 使用Ward方法进行层次聚类
dendrogram(Z)
plt.show()

# 假设labels是聚类算法产生的簇标签
clusters = {}
for i, label in enumerate(labels):
    if label not in clusters:
        clusters[label] = []
    clusters[label].append(i)

# 输出每个簇中的样本
for label, cluster in clusters.items():
    print(f'Cluster {label}: {cluster}')

silhouette_avg = silhouette_score(X, labels)
calinski_harabasz_score = calinski_harabasz_score(X, labels)
print('KMeans-轮廓系数:', silhouette_avg)
print('Kmeans-Calinski-Harabasz Index:', calinski_harabasz_score)

df0 = X[labels == 0].describe()
print(df0)
df0.to_excel('F:\桌面\cluster0.xlsx', index=False)

df1 = X[labels == 1].describe()
print(df1)
df1.to_excel('F:\桌面\cluster1.xlsx', index=False)

df2 = X[labels == 2].describe()
print(df2)
df2.to_excel('F:\桌面\cluster2.xlsx', index=False)



