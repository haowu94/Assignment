import folium
from folium.plugins import MarkerCluster
import pandas as pd

# 加载数据
df = pd.read_csv('../data/spacex_launch_data.csv')
launchpads = pd.read_csv('../data/spacex_launchpads.csv')

# 创建基础地图
world_map = folium.Map(location=[0, 0], zoom_start=2)

# 创建标记簇
marker_cluster = MarkerCluster().add_to(world_map)

# 添加发射场标记
for idx, row in launchpads.iterrows():
    # 解析坐标
    lat, lon = map(float, row['coordinates'].split(', '))

    # 设置标记颜色（基于成功率）
    success_rate = df[df['launch_site'] == row['name']]['success'].mean()
    color = 'green' if success_rate > 0.7 else 'orange' if success_rate > 0.5 else 'red'

    # 创建弹出信息
    popup = f"""
    <b>{row['name']}</b><br>
    Location: {row['location']}<br>
    First Launch: {row['first_launch']}<br>
    Success Rate: {success_rate:.1%}
    """

    # 添加标记
    folium.Marker(
        location=[lat, lon],
        popup=popup,
        icon=folium.Icon(color=color, icon='rocket')
    ).add_to(marker_cluster)

# 添加圆形标记表示覆盖区域
for idx, row in launchpads.iterrows():
    lat, lon = map(float, row['coordinates'].split(', '))
    folium.Circle(
        location=[lat, lon],
        radius=50000,  # 50公里半径
        color='blue',
        fill=True,
        fill_opacity=0.1
    ).add_to(world_map)

# 保存地图
world_map.save('spacex_launch_map.html')