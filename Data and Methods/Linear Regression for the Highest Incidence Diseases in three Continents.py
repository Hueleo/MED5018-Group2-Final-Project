import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load and preview the data
data_path = 'cause_of_deaths.csv'
data = pd.read_csv(data_path)

# Define continent groups (example groups, needs to be adjusted based on actual data)
continent_groups = {
    'Americas': ['United States', 'Brazil', 'Mexico', 'Canada'],
    'Europe': ['Germany', 'United Kingdom', 'France', 'Italy'],
    'Africa': ['Nigeria', 'Egypt', 'South Africa', 'Kenya']  # Added African countries
}

# Find the disease with the highest incidence rate for each continent
top_diseases = {}
for continent, countries in continent_groups.items():
    continent_data = data[data['Country/Territory'].isin(countries)]
    average_deaths = continent_data.iloc[:, 3:].mean().idxmax()
    top_diseases[continent] = average_deaths

# Prepare a plot for linear regression results, now including scatter plots
plt.figure(figsize=(14, 10))

for continent, top_disease in top_diseases.items():
    # Filter data for the specific continent and disease
    continent_data = data[data['Country/Territory'].isin(continent_groups[continent])]
    yearly_deaths = continent_data.groupby('Year')[top_disease].sum().reset_index()

    # Perform linear regression
    X = yearly_deaths[['Year']]
    y = yearly_deaths[top_disease]
    model = LinearRegression()
    model.fit(X, y)
    trend_line = model.predict(X)

    # Plotting the discrete data points as a scatter plot
    plt.scatter(yearly_deaths['Year'], y, label=f'{continent} - {top_disease} Data Points')

    # Plotting the regression line
    plt.plot(yearly_deaths['Year'], trend_line, label=f'{continent} - {top_disease} Trend Line', linestyle='--')

# 防止使用科学计数法
plt.ticklabel_format(style='plain', axis='y')

plt.title('Linear Regression and Data Points of Death Trends for the Highest Incidence Diseases in Continents')
plt.xlabel('Year')
plt.ylabel('Total Deaths')
plt.legend(loc='best', fancybox=True, framealpha=0.5,fontsize='small')  # 'fancybox' 创建圆角图例，'framealpha' 设置透明度
plt.show()
