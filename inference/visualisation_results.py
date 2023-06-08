import os

import dash
import plotly
from dash import Dash, dcc, html, Input, Output, ctx, State
import plotly.express as px
from skimage import io
import plotly.io as pio
import nrrd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import argparse

# Create the parser object
parser = argparse.ArgumentParser(description='Select folder where data is contained.')

# Add the directory argument
parser.add_argument('--folder', '-f', type=str, help='Directory path', default='/home/calde/Desktop/prova')

# Parse the command line arguments
args = parser.parse_args()

# Access the directory argument
directory = args.folder

#%%



path = args.folder
patients = os.listdir(path)

starting_image = os.path.join(path, patients[0], 'image.nrrd')
starting_prediction = os.path.join(path, patients[0], 'prediction_post.nrrd')


#%%

segmentation_mask = nrrd.read(starting_prediction)[0]
ct_scan = nrrd.read(starting_image)[0]
#ground_truth = nrrd.read(f'/home/calde/Desktop/master-thesis-corino/test_predictions/{type}/{patient}/gt_mask_{patient}.nrrd')[0]






#%%

fig = px.imshow(ct_scan[30], zmin=ct_scan.min(), zmax=ct_scan.max(),
                color_continuous_scale='gray')
fig.add_trace(go.Contour(z=segmentation_mask[30], name='unet2.5d', showscale=False, colorscale='turbid',
                         contours=dict(start=0, size=70, end=70, coloring='lines'),
                         line_width=3))

'''fig.add_trace(go.Contour(z=ground_truth[10], showscale=False, name='ground truth', colorscale='Plotly3',
                            contours=dict(start=0, size=70, end=70, coloring='lines'),
                            line_width=3))'''

fig.update_coloraxes(showscale=False)

print(2)

#%%

app = Dash(__name__)

legend_style = {'display': 'flex', 'flex-direction': 'row', 'margin': '10px'}



unet25_square = html.Div(style={'width': '20px', 'height': '20px', 'background-color': 'yellow', 'margin-right': '5px'})
unet25_label = html.Div(children='unet2.5d', style={'margin-right': '10px'})


legend = html.Div(children=[unet25_square, unet25_label], style=legend_style)


# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "10rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

sidebar = html.Div(
    [
        html.H2("Visualization", className="display-4"),
        html.Hr(),
        # select model
        html.H4("Select patient"),
        dcc.Dropdown(id='dropdown', options=list(patients), clearable=False, value=patients[0]),
        #html.H4("Select segmentation masks"),
        #dcc.Checklist(id='checklist', options=['unet2.5d','ground truth'], value=['unet2.5d', 'ground truth']),
    ],
    style=SIDEBAR_STYLE,
)
CONTENT_STYLE = {
    "margin-left": "19rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

content = html.Div([dcc.Graph(id='ct-scan', figure=fig, style={'height': '800px', 'width': '1000px'}),
    legend,
    dcc.Slider(0, len(starting_image), 1,
               value=10,
               id='my-slider',
               updatemode='drag',
               marks=None,
    ),
    html.Div(id='slider-output-container')], style=CONTENT_STYLE)

app.layout = html.Div([sidebar,content])



@app.callback(
    Output('slider-output-container', 'children'),
    Output('ct-scan', 'figure'),
    State('ct-scan', 'figure'),
    Input('my-slider', 'value'),
)

def update_output(fig, value):
    fig = plotly.graph_objects.Figure(fig)

    # change dictionary fig to change z values
    fig['data'][0]['z'] = ct_scan[value]
    fig['data'][1]['z'] = segmentation_mask[value]
    print(value)
    #fig['data'][2]['z'] = ground_truth[value]

    return 'Slice "{}"'.format(value), fig

@app.callback(
    Output('my-slider', 'value'),
    Output('my-slider', 'max'),
    Input('dropdown', 'value'))
def update_output(value):
    global segmentation_mask, ct_scan, seg2, lstm, ground_truth
    ct_scan = nrrd.read(os.path.join(path, value, 'image.nrrd'))[0]
    segmentation_mask = nrrd.read(os.path.join(path, value, 'prediction.nrrd'))[0]
    #ground_truth = nrrd.read(f'/home/calde/Desktop/master-thesis-corino/test_predictions/{patient_type_dic[value]}/{value}/gt_mask_{value}.nrrd')[0]
    #indices = np.load(f'/home/calde/Desktop/master-thesis-corino/test_predictions/{patient_type_dic[value]}/{value}/indices_gt.npy')

    print(value)



    return 10, len(ct_scan) - 1


if __name__ == '__main__':
    app.run_server(debug=True, port=8051)


#%%
