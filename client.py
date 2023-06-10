import requests
import json
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output

def get_data(n):
    url = f"http://127.0.0.1:5000/{n}"
    response = requests.get(url)
    data = json.loads(response.text)
    actions = data['actions']
    prediction = data['prediction']
    return actions, prediction

app = dash.Dash(__name__)

app.layout = html.Div(children=[
    html.H1(children='Модель оптимального управления'),
    html.Label('Количество прогнозируемых дней:'),
    dcc.Input(
        id='n-input',
        type='number',
        value=10
    ),
    html.Button('Обновить', id='submit-button', n_clicks=0),
    html.Div(id='output-container'),
])

@app.callback(
    Output('output-container', 'children'),
    Input('submit-button', 'n_clicks'),
    Input('n-input', 'value')
)
def update_output(n_clicks, n):
    actions, prediction = get_data(n)
    
    prediction_graph = dcc.Graph(
        id='prediction-graph',
        figure={
            'data': [
                go.Scatter(
                    x=list(range(1, n+1)),
                    y=prediction,
                    mode='lines+markers',
                    name='Prediction'
                )
            ],
            'layout': go.Layout(
                title='Прогноз',
                xaxis={'title': 'Прогнозируемые дни'},
                yaxis={'title': 'Значение (млрд. руб)'}
            )
        }
    )

    action_diagrams = []
    for i, action in enumerate(actions):
        action_diagram = dcc.Graph(
            id=f'action-diagram-{i}',
            figure={
                'data': [
                    go.Pie(
                        labels=['1-3', '4-6', '7-12', '13-36'],
                        values=action,
                        hole=0.5
                    )
                ],
                'layout': go.Layout(
                    title=f'Структура портфеля на {i+1} день:',
                )
            }
        )
        action_diagrams.append(action_diagram)

    return [prediction_graph, *action_diagrams]

if __name__ == '__main__':
    app.run_server(debug=False)

