import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# 1、独自研究一下中国的状况
# 得出的结论：

# 加载数据
file_path = 'cause_of_deaths.csv'
data = pd.read_csv(file_path)

# 定义感兴趣的国家
#countries = ['United States',  'Russia',  'United Kingdom']
countries = ['China' ]

# 创建一个图表
plt.figure(figsize=(12, 8))

# 对每个国家进行处理
for country in countries:
    # 筛选出特定国家的数据
    country_data = data[data['Country/Territory'] == country]

    # 找出该国发病率最高的前三个病
    # 计算每种病的平均死亡人数，忽略前两列（国家和年份）
    avg_deaths = country_data.iloc[:, 2:].mean().sort_values(ascending=False)
    top_diseases = avg_deaths.head(3).index  # 获取发病率最高的病

    # 对这些病进行线性回归分析
    for disease in top_diseases:
        # 选择因变量和自变量
        Y = country_data[disease]  # 特定病的死亡人数作为响应变量
        X = country_data[['Year']]  # 年份作为解释变量

        # 添加常数项
        X = sm.add_constant(X)

        # 构建模型并拟合数据
        model = sm.OLS(Y, X).fit()

        # 获取R^2值和回归系数的p值
        r_squared = model.rsquared
        p_value = model.pvalues[1]  # Year的系数p值

        # 绘制散点图
        plt.scatter(country_data['Year'], Y, label=f'{country} - {disease}')

        # 计算回归线的预测值并绘制
        predictions = model.predict(X)
        plt.plot(country_data['Year'], predictions, label=f'{country} {disease} Regression (R²={r_squared:.2f}, p={p_value:.2e})')

# 防止使用科学计数法
plt.ticklabel_format(style='plain', axis='y')



# 添加标签和标题
plt.xlabel('Year')
plt.ylabel('Deaths')
plt.title('Top 3 Diseases Death Over Time in China')
plt.legend(loc='best', fancybox=True, framealpha=0.5,fontsize='small')  # 'fancybox' 创建圆角图例，'framealpha' 设置透明度



# 显示图表
plt.show()
