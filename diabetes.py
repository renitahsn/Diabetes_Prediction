import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import plotly.graph_objs as go
import pandas as pd
import seaborn as sns
import dash_table
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
import pickle

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,roc_auc_score, f1_score, log_loss, matthews_corrcoef

def generate_table(dataframe, page_size=10):
    return dash_table.DataTable(
        id = 'dataTable',
        columns = [{
            "name": i, 
            "id": i
        } for i in dataframe.columns],
        data = dataframe.to_dict('records'),
        page_action = "native",
        page_current = 0,
        page_size = page_size
    )

rfc = pickle.load(open('Randomforest.sav','rb'))
ada = pickle.load(open('adaboost.sav','rb'))
logit = pickle.load(open('logistic.sav','rb'))
knn = pickle.load(open('knn.sav','rb'))

diabetes = pickle.load(open('diabetes.sav', 'rb'))
dataframe = pd.DataFrame(diabetes)
external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
    html.H1("Hello Dash"),
    html.Div(children="""Data of Diabetes People in PIMA."""),
    dcc.Tabs(children = [
        dcc.Tab(value = 'Tab1', label = 'Dataframe Table', children = [
            html.Div([generate_table(dataframe, page_size=10)])
        ]),
        dcc.Tab(label = 'Scatter-Chart', value = 'tab2', children =[
                html.Div(children = dcc.Graph(
                id = 'graph-scatter',
                figure = {'data': [
                    go.Scatter(
                        x = (diabetes[diabetes['Outcome'] == i]['Glucose']),
                        y = (diabetes[diabetes['Outcome'] == i]['Age']),
                        text = diabetes[diabetes['Outcome'] == i]['Glucose_test'],
                        mode='markers',
                        name = '{}'.format(i)
                        )for i in diabetes['Outcome'].unique()
                ],
                    'layout':go.Layout(
                        xaxis= {'title': 'Glucose'},
                        yaxis={'title': 'Age'},
                        hovermode='closest'
                    )
                }
            )),
            ]),

        dcc.Tab(label = 'Bar-Chart', value = 'tab3', children=
             [html.Div(children =[
                html.Div(children =[
                html.P('Y1:'),    
                dcc.Dropdown(id = 'y-axis-1', options = [{'label': i, 'value': i} for i in diabetes.select_dtypes('number').columns], 
                value = 'BMI')
                ], className = 'col-3'),
                html.Div(children =[
                html.P('X:'),    
                dcc.Dropdown(id = 'x-axis-1', options = [{'label': i, 'value': i} for i in ['Insulin', 'Glucose', 'SkinThickness']], 
                value = 'Outcome')
                ], className = 'col-3')    
                ], className = 'row'),

                html.Div([
                ## Graph Bar
                dcc.Graph(
                id = 'graph-bar',
                figure ={
                    'data' : [
                        {'x': diabetes['Outcome'], 'y': diabetes['BMI'], 'type': 'bar', 'name' :'BMI'}
                    ], 
                    'layout': {'title': 'Bar Chart'}  
                    }
                    )])]),

        dcc.Tab(label = 'Prediction Machine Learning', value = 'tab4', children=
            [html.Div([
                html.Div([
                html.P('Please Select Model'),
                dcc.Dropdown(value = 'None',
                            id = 'filter_model',
                             options =[
                                    {'label' : 'Random Forest', 'value' : 'rfc'},
                                    {'label' : 'AdaBoost', 'value' : 'ada'},
                                    {'label' : 'Logistic Regression', 'value' : 'logit'},
                                    {'label' : 'KNN', 'value' : 'knn'},
                                    {'label' : 'None', 'value' : 'None'}
                             ]
                )
                ], className = 'col-3'),], className='row'),

                html.Br(),
                html.Div(children = [
                    html.Button('Check Accuracy', id='get_accuracy')
                ], className='row col-3'),

                html.Br(),
                html.Div(id='output_prediction')], className = 'col-3'),
    

    ], ## Tabs Content Style
        content_style = {
            'fontFamiliy': 'Arial',
            'borderBottom': '1px solid #d6d6d6',
            'borderLeft': '1px solid #d6d6d6',
            'borderRight': '1px solid #d6d6d6',
            'padding': '44px'}
        )
],  #Div paling luar
    style = {
        'maxWidth': '1200px',
        'margin': '0 auto'
    }
)

@app.callback(
    Output(component_id = 'graph-bar', component_property = 'figure'),
    [Input(component_id = 'y-axis-1', component_property = 'value'),
    Input(component_id = 'x-axis-1', component_property = 'value'),]
)
def create_graph_bar(y1, x1):
    figure = {
                    'data' : [
                        {'x': diabetes[x1], 'y': diabetes[y1], 'type': 'bar', 'name' :y1}
                    ], 
                    'layout': {'title': 'Bar Chart'}  
                    }
    return figure     
       
@app.callback(
    Output(component_id='output_prediction', component_property='children'),
    [Input(component_id='get_accuracy', component_property='n_clicks')],
    [State(component_id='filter_model', component_property='value')])

def evaluasi_model(n_clicks, filter_model):
    if (filter_model == '') or (filter_model =='Null') or (filter_model == 'None'):
        return ''
    else:
        x = pickle.load(open('x.sav','rb'))
        y = pickle.load(open('y.sav','rb'))
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)

        result = 0.0
        if filter_model =='rfc':
            rfc.fit(X_train,y_train)
            result = accuracy_score(y_test,rfc.predict(X_test))
        elif filter_model == 'ada':
            ada.fit(X_train,y_train)
            result = accuracy_score(y_test,ada.predict(X_test))
        elif filter_model == 'logit':
            logit.fit(X_train,y_train)
            result = accuracy_score(y_test,logit.predict(X_test))
        elif filter_model == 'knn':
            knn.fit(X_train,y_train)
            result = accuracy_score(y_test,knn.predict(X_test))

        return 'Accuracy value from the model = {}'.format(result)

if __name__ == '__main__':
    app.run_server(debug=True, port=1997)