"""
This app creates a simple sidebar layout using inline style arguments and the
dbc.Nav component.

dcc.Location is used to track the current location, and a callback uses the
current location to render the appropriate page content. The active prop of
each NavLink is set automatically according to the current pathname. To use
this feature you must install dash-bootstrap-components >= 0.11.0.

For more details on building multi-page Dash applications, check out the Dash
documentation: https://dash.plot.ly/urls
"""
import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html
import pandas as pd
import plotly.express as px
import os
from natsort import natsorted

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

df = pd.read_csv('../data/csv_files/val_evaluation.csv')

# get all unique patients as a list
patients = df['patient'].unique().tolist()

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "20rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "22rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

sidebar = html.Div(
    [
        html.H2("Evaluations", className="display-4"),
        html.Hr(),
        html.H6("Select patient"),
        dcc.Dropdown(id='dropdown', options=patients, value=''),
    ],
    style=SIDEBAR_STYLE,
)
{'src': f'assets/test/imagesAM01.png'}
carousel = dbc.Carousel(
    id='carousel',
    items=[],
    controls=True,
    indicators=True,
    variant='dark',
    className="carousel-fade",
    style={"height": "1100px", "width": "1100px", 'margin_left':'27rem'}  # set th

)

content = html.Div([dcc.Graph(id='bar-chart-1'),
                    carousel], style=CONTENT_STYLE)

app.layout = html.Div([dcc.Location(id="url"), sidebar, content])


@app.callback(
    Output("bar-chart-1", "figure"),
    Input("dropdown", "value"),
)
def render_bar_chart(patient):
    # filter the dataframe to get only the rows for the chosen patient
    patient_df = df[df['patient'] == patient]

    fig = px.bar(patient_df, x='serial_number', y='iou', orientation='v', title=f'IoU for patient : {patient}',
                 color='iou')
    # update the layout of the figure. y axis is from 0 to 1
    fig.update_layout(yaxis_range=[0, 1])

    return fig




@app.callback(
    Output("carousel", "items"),
    Input("dropdown", "value"),
)
def render_carousel(patient):
    if patient is None or patient == '':
        return []
    path = os.path.join('assets', patient)
    images = os.listdir(path)
    images = [os.path.join(path, image) for image in natsorted(images)]
    print(images)
    items = [{'src': image, 'height': 20, 'width': 100} for image in images]
    return items



if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8135)
