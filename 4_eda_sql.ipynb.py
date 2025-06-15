import sqlite3
import pandas as pd

# 创建内存数据库
conn = sqlite3.connect(':memory:')
df = pd.read_csv('../data/spacex_launch_data.csv')
df.to_sql('launches', conn, index=False)

# SQL查询函数
def run_sql_query(query):
    return pd.read_sql_query(query, conn)

# 1. 唯一发射场
unique_sites = run_sql_query("SELECT DISTINCT launch_site FROM launches")
print("Unique Launch Sites:\n", unique_sites)

# 2. 5个CCA开头发射场
cca_sites = run_sql_query("SELECT * FROM launches WHERE launch_site LIKE 'CCA%' LIMIT 5")
print("\nLaunch Sites starting with 'CCA':\n", cca_sites)

# 3. NASA总有效载荷
nasa_payload = run_sql_query("SELECT SUM(payload_mass) FROM launches")
print("\nTotal Payload Mass:", nasa_payload.iloc[0,0])

# 4. 首次成功地面着陆
first_success = run_sql_query(
    "SELECT MIN(date) FROM launches WHERE landing_outcome = 'Success (ground pad)'"
)
print("\nFirst Successful Ground Landing:", first_success.iloc[0,0])

# 5. 有效载荷在4000-6000kg的成功无人机船着陆
drone_success = run_sql_query(
    "SELECT booster_name FROM launches "
    "WHERE payload_mass BETWEEN 4000 AND 6000 "
    "AND landing_outcome = 'Success (drone ship)'"
)
print("\nSuccessful Drone Ship Landings (4000-6000kg):\n", drone_success)

# 保存所有查询结果
all_queries = [
    ("Unique Launch Sites", "SELECT DISTINCT launch_site FROM launches"),
    # ... 添加所有14个查询
]

for name, query in all_queries:
    result = run_sql_query(query)
    result.to_csv(f'sql_results/{name.replace(" ", "_")}.csv', index=False)