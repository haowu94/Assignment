import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd

# 加载数据
df = pd.read_csv('../data/spacex_launch_data.csv')

# 创建Dash应用
app = dash.Dash(__name__)

# 应用布局
app.layout = html.Div([
    html.H1("SpaceX Launch Success Dashboard"),

    html.Div([
        dcc.Dropdown(
            id='site-dropdown',
            options=[{'label': 'All Sites', 'value': 'ALL'}] +
                    [{'label': site, 'value': site} for site in df['launch_site'].unique()],
            value='ALL',
            placeholder="Select a Launch Site"
        ),
    ]),

    html.Div([
        dcc.RangeSlider(
            id='payload-slider',
            min=0,
            max=10000,
            step=1000,
            marks={i: f'{i} kg' for i in range(0, 10001, 2000)},
            value=[0, 10000]
        )
    ]),

    html.Div([
        dcc.Graph(id='success-pie'),
        dcc.Graph(id='payload-scatter')
    ], style={'display': 'flex'}),
])


# 回调函数
@app.callback(
    Output('success-pie', 'figure'),
    Input('site-dropdown', 'value')
)
def update_pie(selected_site):
    if selected_site == 'ALL':
        fig = px.pie(df, names='success', title='Overall Success Rate')
    else:
        filtered_df = df[df['launch_site'] == selected_site]
        fig = px.pie(filtered_df, names='success',
                     title=f'Success Rate at {selected_site}')
    return fig


@app.callback(
    Output('payload-scatter', 'figure'),
    [Input('site-dropdown', 'value'), Input('payload-slider', 'value')]
)
def update_scatter(selected_site, payload_range):
    low, high = payload_range
    filtered_df = df[(df['payload_mass'] >= low) & (df['payload_mass'] <= high)]

    if selected_site != 'ALL':
        filtered_df = filtered_df[filtered_df['launch_site'] == selected_site]

    fig = px.scatter(
        filtered_df,
        x='payload_mass',
        y='flight_number',
        color='success',
        title=f'Payload vs. Flight Outcome',
        labels={'payload_mass': 'Payload Mass (kg)', 'flight_number': 'Flight Number'}
    )
    return fig


# 运行应用
if __name__ == '__main__':
    app.run_server(debug=True)