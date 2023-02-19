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
cwd = os.getcwd()

# loop inside assets folder and get all the name of the folders
# these folders are the patients
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
        # select model
        html.H6("Select model"),
        dcc.Dropdown(id='dropdown_model', options=os.listdir(os.path.join(cwd, 'assets', 'models_evaluation')), value=''),
        html.H6("Select patient"),
        dcc.Dropdown(id='dropdown', options=[], value=''),
    ],
    style=SIDEBAR_STYLE,
)


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
    Output("dropdown", "options"),
    Input("dropdown_model", "value"),
)
def update_dropdown(model_name):
    if model_name is None or model_name == '':
        return []
    df = pd.read_csv(f'../evaluation/csv_files/{model_name}.csv')
    # get all unique patients as a list
    patients = df['patient'].unique().tolist()
    # create a list of dictionaries with the patient name and the patient name
    options = [{'label': patient, 'value': patient} for patient in patients]
    return options

@app.callback(
    Output("bar-chart-1", "figure"),
    Input("dropdown", "value"),
    Input("dropdown_model", "value"),
)
def render_bar_chart(patient, model_name):
    if patient is None or patient == '':
        return dash.no_update
    df = pd.read_csv(f'../evaluation/csv_files/{model_name}.csv')

    # get all unique patients as a list
    patients = df['patient'].unique().tolist()
    # filter the dataframe to get only the rows for the chosen patient
    patient_df = df[df['patient'] == patient]

    fig = px.bar(patient_df,
                 x='serial_number',
                 y=["iou", "dice"],
                 barmode='group',
                 title=f'IoU for patient : {patient}',)
                 #color='iou')
    # update the layout of the figure. y axis is from 0 to 1
    fig.update_layout(yaxis_range=[0, 1])

    return fig




@app.callback(
    Output("carousel", "items"),
    Output("carousel", "active_index"),
    Input("dropdown", "value"),
    Input("dropdown_model", "value"),
)
def render_carousel(patient, model_name):
    if patient is None or patient == '':
        return [], 0
    path = os.path.join('assets', 'models_evaluation', model_name, patient)
    images = os.listdir(path)
    images = [os.path.join(path, image) for image in natsorted(images)]
    print(images)
    items = [{'src': image, 'height': 20, 'width': 100} for image in images]
    return items, 0



if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8137)
