import requests
import pandas as pd
from bs4 import BeautifulSoup


# 通过SpaceX API获取数据
def get_spacex_launches():
    url = "https://api.spacexdata.com/v4/launches"
    response = requests.get(url)
    launches = response.json()

    # 提取关键字段
    data = []
    for launch in launches:
        flight_data = {
            'flight_number': launch['flight_number'],
            'date': launch['date_utc'],
            'rocket': launch['rocket'],
            'success': launch['success'],
            'launchpad': launch['launchpad'],
            'payloads': launch['payloads'],
            'details': launch['details']
        }
        data.append(flight_data)

    return pd.DataFrame(data)


# 从维基百科抓取发射场数据
def scrape_launchpads():
    url = "https://en.wikipedia.org/wiki/List_of_SpaceX_launch_sites"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # 解析表格数据
    table = soup.find('table', {'class': 'wikitable'})
    rows = table.find_all('tr')[1:]  # 跳过表头

    launchpads = []
    for row in rows:
        cols = row.find_all('td')
        if len(cols) >= 4:
            launchpads.append({
                'name': cols[0].text.strip(),
                'location': cols[1].text.strip(),
                'coordinates': cols[2].text.strip(),
                'first_launch': cols[3].text.strip()
            })

    return pd.DataFrame(launchpads)


# 主函数
if __name__ == "__main__":
    # 获取数据
    launches_df = get_spacex_launches()
    launchpads_df = scrape_launchpads()

    # 保存数据
    launches_df.to_csv('../data/spacex_launch_data_raw.csv', index=False)
    launchpads_df.to_csv('../data/spacex_launchpads_raw.csv', index=False)

    print("done！")