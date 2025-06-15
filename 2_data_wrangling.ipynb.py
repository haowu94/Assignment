import pandas as pd
import numpy as np

# 加载原始数据
launches = pd.read_csv('../data/spacex_launch_data_raw.csv')
launchpads = pd.read_csv('../data/spacex_launchpads_raw.csv')


# 数据清洗函数
def clean_launch_data(df):
    # 处理缺失值
    df['payload_mass'] = df['payloads'].apply(
        lambda x: sum([p['mass_kg'] for p in x]) if x and isinstance(x, list) else np.nan
    )
    df['payload_mass'].fillna(df['payload_mass'].median(), inplace=True)

    # 转换日期格式
    df['date'] = pd.to_datetime(df['date'])
    df['launch_year'] = df['date'].dt.year

    # 简化成功列
    df['success'] = df['success'].astype(int)

    # 合并发射场名称
    launchpad_names = launchpads.set_index('id')['name'].to_dict()
    df['launch_site'] = df['launchpad'].map(launchpad_names)

    # 选择关键特征
    features = [
        'flight_number', 'date', 'launch_year', 'rocket',
        'launch_site', 'payload_mass', 'success'
    ]
    return df[features]


# 主函数
if __name__ == "__main__":
    cleaned_launches = clean_launch_data(launches)
    cleaned_launches.to_csv('../data/spacex_launch_data.csv', index=False)
    print("done！")