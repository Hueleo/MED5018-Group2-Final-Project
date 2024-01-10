import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

#分析角度：全球死因的时间趋势
#目标：
#分析时间跨度内全球或大区域（如非洲、亚洲、欧洲等）的主要死因趋势。
#了解哪些疾病的死亡人数呈上升或下降趋势。
#评估时间的影响力度，即随着时间的推移，死亡率的变化速率。

# 加载数据
file_path = 'cause_of_deaths.csv'
data = pd.read_csv(file_path)

# 聚合数据以计算每种病因每年的全球总死亡人数
global_deaths_per_year = data.iloc[:, 3:].groupby(data['Year']).sum()

# 选择全球性的主要死因进行分析
diseases_to_analyze = global_deaths_per_year.mean().sort_values(ascending=False).head(3).index

# 创建一个图表
plt.figure(figsize=(12, 8))

# 分析每种疾病的时间趋势
for disease in diseases_to_analyze:
    # 选择因变量和自变量
    Y = global_deaths_per_year[disease]  # 死亡人数作为响应变量
    X = global_deaths_per_year.index.to_list()  # 年份作为解释变量
    X = sm.add_constant(X)  # 添加常数项

    # 构建模型并拟合数据
    model = sm.OLS(Y, X).fit()

    # 获取R^2值和回归系数的p值
    r_squared = model.rsquared
    p_value = model.pvalues[1]  # Year的系数p值

    # 绘制散点图
    plt.scatter(global_deaths_per_year.index, Y, label=f'{disease}')

    # 计算回归线的预测值并绘制
    predictions = model.predict(X)
    plt.plot(global_deaths_per_year.index, predictions, label=f'{disease} Trend (R²={r_squared:.2f}, p={p_value:.2e})')

# 防止使用科学计数法
plt.ticklabel_format(style='plain', axis='y')

# 添加标签和标题
plt.xlabel('Year')
plt.ylabel('Global Deaths')
plt.title('Global Deaths Over Time for Top Diseases')
plt.legend(loc='upper left', fancybox=True, framealpha=0.5,fontsize='medium')  # 'fancybox' 创建圆角图例，'framealpha' 设置透明度

# 显示图表
plt.show()
