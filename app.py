import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from dash import dash_table
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score, GridSearchCV
import plotly.express as px
import plotly.graph_objects as go

# Load the dataset
df = pd.read_csv('cancer ds.csv')

# Select relevant columns and drop rows with missing values
selected_columns = ['avganncount', 'avgdeathsperyear', 'target_deathrate', 'incidencerate', 'medincome', 'povertypercent', 'medianage']
data = df[selected_columns].dropna()

# Standardize the data
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data)

# Split the data into features and target
X = data_normalized
y = [1] * len(X)  # All instances are considered inliers initially

# Initialize and fit the Isolation Forest model
model = IsolationForest(contamination=0.05, random_state=42)
model.fit(X)

# Predict anomalies on the whole dataset
y_pred = model.predict(X)
y_pred = [1 if x == 1 else -1 for x in y_pred]

# Calculate Precision, Recall, and F1 Score
precision = precision_score(y, y_pred, pos_label=1)
recall = recall_score(y, y_pred, pos_label=1)
f1 = f1_score(y, y_pred, pos_label=1)

# Perform cross-validation to evaluate model performance
cv_scores = cross_val_score(model, X, y, cv=5, scoring='f1_macro')

# Perform grid search for hyperparameter tuning
param_grid = {'contamination': [0.01, 0.05, 0.1]}
grid_search = GridSearchCV(estimator=IsolationForest(random_state=42), param_grid=param_grid, scoring='f1_macro', cv=5)
grid_search.fit(X, y)

# Correlation matrix
correlation_matrix = data.corr()

# Number of anomalies detected
num_anomalies = sum([1 for x in y_pred if x == -1])

# Feature importance analysis
feature_importances = model.decision_function(X)

# Convert decision function scores for visualization
important_features = pd.Series(feature_importances, index=range(len(feature_importances)))
important_features.sort_values(ascending=False, inplace=True)

# Create Dash application
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])

# Layout of the Dash application
app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1("Cancer Dataset Anomaly Detection Dashboard", className="text-center text-primary my-4"))),
    
    dbc.Row([
        dbc.Col([
            html.H2("Basic Statistics", className="text-primary"),
            dash_table.DataTable(
                data=data.describe().reset_index().to_dict('records'),
                columns=[{'name': i, 'id': i} for i in data.describe().reset_index().columns],
                style_table={'overflowX': 'auto'},
                style_header={'backgroundColor': '#f2f2f2', 'fontWeight': 'bold'},
                style_data={'backgroundColor': 'rgb(248, 248, 248)', 'fontWeight': 'bold'}
            )
        ])
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            html.H2("Feature Distributions", className="text-primary"),
            dcc.Dropdown(
                id='feature-dropdown',
                options=[{'label': col, 'value': col} for col in selected_columns],
                value=selected_columns[0],
                style={'color': 'black'}
            ),
            dcc.Graph(id='feature-boxplot')
        ])
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            html.H2("Correlation Matrix", className="text-primary"),
            dcc.Graph(
                figure=px.imshow(correlation_matrix, text_auto=True, color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
            )
        ])
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Model Performance Metrics", className="text-white bg-primary font-weight-bold"),
                dbc.CardBody([
                    html.P(f"Precision: {precision}", className="card-text"),
                    html.P(f"Recall: {recall}", className="card-text"),
                    html.P(f"F1 Score: {f1}", className="card-text"),
                    html.P(f"Cross-Validation F1 Scores: {cv_scores}", className="card-text"),
                    html.P(f"Best Contamination Parameter: {grid_search.best_params_['contamination']}", className="card-text"),
                    html.P(f"Number of anomalies detected: {num_anomalies}", className="card-text")
                ])
            ], outline=True)
        ])
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            html.H2("Anomaly Detection - Decision Function Scores", className="text-primary"),
            dcc.Graph(
                figure=go.Figure(data=go.Scatter(
                    x=list(range(len(X))),
                    y=feature_importances,
                    mode='markers',
                    marker=dict(color=y_pred, colorscale='RdBu')
                )).update_layout(title='Anomaly Detection - Decision Function Scores',
                                 xaxis_title='Sample Index',
                                 yaxis_title='Decision Function Score',
                                 coloraxis_colorbar=dict(title='Anomaly (-1) or Normal (1)'))
            )
        ])
    ])
], fluid=True)

# Callback for updating feature box plot
@app.callback(
    Output('feature-boxplot', 'figure'),
    Input('feature-dropdown', 'value')
)
def update_boxplot(selected_feature):
    try:
        if selected_feature not in data.columns:
            return go.Figure()

        fig = px.box(data, y=selected_feature, title=f'Distribution of {selected_feature}')
        return fig
    except Exception as e:
        print(f"Error in update_boxplot: {e}")
        return go.Figure()

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
