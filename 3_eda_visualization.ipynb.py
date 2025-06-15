import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 加载清洗后的数据
df = pd.read_csv('../data/spacex_launch_data.csv')

# 1. 发射次数与发射场关系
plt.figure(figsize=(10, 6))
sns.scatterplot(x='flight_number', y='launch_site', data=df, hue='success')
plt.title('Flight Number vs. Launch Site')
plt.xlabel('Flight Number')
plt.ylabel('Launch Site')
plt.savefig('flight_vs_site.png')
plt.show()

# 2. 有效载荷与发射场关系
plt.figure(figsize=(10, 6))
sns.scatterplot(x='payload_mass', y='launch_site', data=df, hue='success')
plt.title('Payload Mass vs. Launch Site')
plt.xlabel('Payload Mass (kg)')
plt.ylabel('Launch Site')
plt.savefig('payload_vs_site.png')
plt.show()

# 3. 轨道类型成功率
orbit_success = df.groupby('orbit')['success'].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.barplot(x='orbit', y='success', data=orbit_success)
plt.title('Success Rate by Orbit Type')
plt.xlabel('Orbit Type')
plt.ylabel('Success Rate')
plt.savefig('orbit_success.png')
plt.show()

# 4. 年度成功趋势
yearly_success = df.groupby('launch_year')['success'].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.lineplot(x='launch_year', y='success', data=yearly_success)
plt.title('Yearly Launch Success Trend')
plt.xlabel('Year')
plt.ylabel('Success Rate')
plt.savefig('yearly_trend.png')
plt.show()