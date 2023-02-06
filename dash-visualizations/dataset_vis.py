import os

from dash import Dash ,html, dcc
import pandas as pd
import plotly.express as px

app = Dash(__name__)

# Create sample data
df = pd.read_csv('../data/csv_files/whole_dataset.csv')
# group dataframe by category and count the number of rows with distinct patients
patients_per_category = df.groupby('category')['patient'].nunique().reset_index(name='counts')
images_per_patient = df.groupby(['patient', 'category']).size().reset_index(name='counts')
images_per_patient = images_per_patient.sort_values(by='counts', ascending=True)
images_per_category = df.groupby(['category']).size().reset_index(name='counts')
total_rows = df.shape[0]



# Plot the first bar chart
fig1 = px.bar(patients_per_category, x='category', y='counts', text='counts')
fig1.update_layout(title={'text': "Patients per Category", 'font': {'size': 30}},
                   xaxis_title='Category',
                   title_x=0.5,
                   yaxis_title='# Patients',
                   font=dict(size=15))

# create pie chart for images per category
fig3 = px.pie(images_per_category, values='counts', names='category', title='Images per category')
fig3.update_layout(title={'text': "Images per category", 'font': {'size': 30}},
                   title_x=0.5,
                   font=dict(size=15))





# Plot the second bar chart
fig2 = px.bar(images_per_patient, x='counts', y='patient', orientation='h', title='Images per patient', color='counts',
              height=1600, width=1600)
# add a vertical line to show the mean
fig2.add_vline(x=images_per_patient['counts'].mean(), line_width=3, line_dash="dash", line_color="green")
# add number of mean to vertical line
fig2.add_annotation(x=images_per_patient['counts'].mean(), y=0.5, text="Mean: " + str(round(images_per_patient['counts'].mean(), 2)), showarrow=False)
#
fig2.update_layout(title={'text': f"Images per patient. (Total images : {total_rows})", 'font': {'size': 40}},
                   yaxis_title='Patients',
                   font=dict(size=15))

# update layout. put title in the middle
fig2.update_layout(title_x=0.5)


app.layout = html.Div(style={}, children=[
    html.H1(children='Dataset statistics', style={'textAlign': 'center', 'marginTop': 50}),
    html.Div([
        dcc.Graph(id='bar-chart-1', figure=fig1),
        dcc.Graph(id='pie-chart-1', figure=fig3)
    ], style={'display': 'flex', 'justifyContent': 'center', 'marginTop': 50}),
    html.Div([
        dcc.Graph(id='bar-chart-2', figure=fig2),
    ], style={'display': 'flex', 'justifyContent': 'center', 'marginTop': 50}),

    # add video below the other two graphs and center in the page

    html.H1(children='Images visualization', style={'textAlign': 'center', 'marginTop': 50}),

    html.Video(src='assets/comparison.mp4', controls=True, style={'display': 'block', 'margin-left': 'auto', 'margin-right': 'auto'}, title='Preprocessing comparison'),
    html.Video(src='assets/comparison-just-masks.mp4', controls=True, style={'display': 'block', 'margin-left': 'auto', 'margin-right': 'auto'}),
])
'''app.layout = html.Div([
    html.H1(children='Two Bar Charts Example', style={'textAlign': 'center'}),
    html.Div([
        dcc.Graph(id='bar-chart-1', figure=fig1),
        dcc.Graph(id='bar-chart-2', figure=fig2)
    ], style={'display': 'flex', 'justifyContent': 'center'})
])'''

if __name__ == '__main__':
    print(os.getcwd())
    app.run_server(debug=True)
