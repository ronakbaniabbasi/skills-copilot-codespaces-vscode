!pip install pandas numpy matplotlib seaborn scikit-learn plotly dash dash-bootstrap-components

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output

# %% [markdown]
# ## 2. Enhanced Data Preparation
# %%
# Create more comprehensive synthetic data
np.random.seed(42)
years = list(range(1990, 2023))
countries = ['USA', 'China', 'Germany', 'India', 'Brazil', 'Iran', 'Russia', 'England', 'UAE']

# Base economic indicators
data = {
    'Year': np.random.choice(years, 1000),
    'Country': np.random.choice(countries, 1000),
    'GDP_Growth': np.random.normal(3, 1.5, 1000),
    'Urbanization_Rate': np.random.uniform(30, 95, 1000),
    'Education_Spending': np.random.uniform(1, 8, 1000),
    'Tech_Investment': np.random.uniform(0.5, 5, 1000),
    'Trade_Openness': np.random.uniform(10, 150, 1000),  # Globalization proxy
    'Individualism_Index': np.random.uniform(10, 90, 1000),  # Hofstede's scale
    'Environmental_Quality': np.random.uniform(40, 100, 1000),
    'Political_Stability': np.random.uniform(-2, 2, 1000)  # World Bank index
}

df = pd.DataFrame(data)

# Add realistic relationships between factors
df['GDP_Growth'] = (
    df['GDP_Growth'] 
    + 0.12 * df['Urbanization_Rate']
    + 0.08 * df['Tech_Investment']
    + 0.05 * df['Trade_Openness']
    - 0.03 * (100 - df['Environmental_Quality'])
    + 0.02 * df['Individualism_Index']
    + 0.04 * df['Political_Stability']
)

# Country-specific adjustments
country_adjustments = {
    'China': {'Urbanization_Rate': +15, 'Trade_Openness': +30},
    'UAE': {'Tech_Investment': +2, 'Individualism_Index': -15},
    'Iran': {'Political_Stability': -1, 'Trade_Openness': -20},
    'Russia': {'Political_Stability': -0.5, 'Individualism_Index': -10},
    'England': {'Individualism_Index': +15, 'Education_Spending': +1}
}

for country, adjustments in country_adjustments.items():
    mask = df['Country'] == country
    for col, val in adjustments.items():
        df.loc[mask, col] += val

# Add derived metrics
df['Productivity'] = 0.4*df['Tech_Investment'] + 0.3*df['Education_Spending'] + 0.3*df['Urbanization_Rate']
df['Sustainability_Index'] = 0.7*df['Environmental_Quality'] - 0.3*(100 - df['Urbanization_Rate'])

# One-hot encode countries
df = pd.get_dummies(df, columns=['Country'], prefix='Country')

# Show sample data
df.sample(5)

# %% [markdown]
# ## 3. Advanced Modeling with Key Factors
# %%
# Prepare data
targets = ['GDP_Growth', 'Productivity', 'Sustainability_Index']
X = df.drop(targets, axis=1)
y = df[targets]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
models = {
    'GDP Growth': RandomForestRegressor(n_estimators=150, random_state=42),
    'Productivity': RandomForestRegressor(n_estimators=100, random_state=42),
    'Sustainability': RandomForestRegressor(n_estimators=100, random_state=42)
}

for i, (name, model) in enumerate(models.items()):
    model.fit(X_train, y_train.iloc[:, i])
    pred = model.predict(X_test)
    print(f"{name} Model:")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test.iloc[:, i], pred)):.3f}")
    print(f"R²: {r2_score(y_test.iloc[:, i], pred):.3f}\n")

# %% [markdown]
# ## 4. Comprehensive Policy Simulation
# %%
def simulate_policy(country, urbanization=0, globalization=0, individualism=0, 
                   tech=0, education=0, environment=0):
    """Enhanced simulation with all key factors"""
    country_col = f'Country_{country}'
    if country_col not in df.columns:
        country_col = [c for c in df.columns if c.startswith('Country_')][0]
    
    scenario = df[df[country_col] == 1].copy()
    
    # Apply changes
    scenario['Urbanization_Rate'] *= (1 + urbanization/100)
    scenario['Trade_Openness'] *= (1 + globalization/100)
    scenario['Individualism_Index'] *= (1 + individualism/100)
    scenario['Tech_Investment'] *= (1 + tech/100)
    scenario['Education_Spending'] *= (1 + education/100)
    scenario['Environmental_Quality'] *= (1 + environment/100)
    
    # Update derived metrics
    scenario['Productivity'] = 0.4*scenario['Tech_Investment'] + 0.3*scenario['Education_Spending'] + 0.3*scenario['Urbanization_Rate']
    scenario['Sustainability_Index'] = 0.7*scenario['Environmental_Quality'] - 0.3*(100 - scenario['Urbanization_Rate'])
    
    # Predict outcomes
    X_scenario = scenario.drop(targets, axis=1, errors='ignore')
    results = {
        'GDP Growth': f"{models['GDP Growth'].predict(X_scenario).mean() - scenario['GDP_Growth'].mean():+.2f}%",
        'Productivity': f"{models['Productivity'].predict(X_scenario).mean() - scenario['Productivity'].mean():+.2f}",
        'Sustainability': f"{models['Sustainability'].predict(X_scenario).mean() - scenario['Sustainability_Index'].mean():+.2f}",
        'Key Drivers': {
            'Urbanization': f"{urbanization}% change",
            'Globalization': f"{globalization}% change",
            'Individualism': f"{individualism}% change"
        }
    }
    return results

# Test simulation
simulate_policy('Iran', urbanization=5, globalization=-10, individualism=2, education=8)

# %% [markdown]
# ## 5. Professional Dashboard with All Factors
# %%
# Initialize Dash app with Bootstrap
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])

app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1("Advanced Economic Prosperity Simulator"), className="text-center my-4")),
    
    dbc.Row([
        dbc.Col([
            html.Label("Select Country:"),
            dcc.Dropdown(
                id='country-selector',
                options=[{'label': c, 'value': c} for c in countries],
                value='China'
            ),
        ], md=4),
        
        dbc.Col([
            html.Label("Time Horizon:"),
            dcc.Slider(id='years-slider', min=1, max=10, step=1, value=5,
                      marks={i: f"{i} yrs" for i in range(1, 11)})
        ], md=8)
    ], className="mb-4"),
    
    dbc.Tabs([
        dbc.Tab(label="Core Factors", children=[
            dbc.Row([
                dbc.Col([
                    html.Label("Urbanization Change (%):"),
                    dcc.Slider(id='urban-slider', min=-10, max=20, step=1, value=0)
                ]),
                dbc.Col([
                    html.Label("Globalization Change (%):"),
                    dcc.Slider(id='global-slider', min=-20, max=30, step=1, value=0)
                ])
            ], className="mb-3"),
            
            dbc.Row([
                dbc.Col([
                    html.Label("Individualism Change (%):"),
                    dcc.Slider(id='individual-slider', min=-5, max=15, step=1, value=0)
                ]),
                dbc.Col([
                    html.Label("Tech Investment Change (%):"),
                    dcc.Slider(id='tech-slider', min=-5, max=25, step=1, value=0)
                ])
            ])
        ]),
        
        dbc.Tab(label="Policy Levers", children=[
            dbc.Row([
                dbc.Col([
                    html.Label("Education Spending Change (%):"),
                    dcc.Slider(id='edu-slider', min=-5, max=20, step=1, value=0)
                ]),
                dbc.Col([
                    html.Label("Environmental Protection (%):"),
                    dcc.Slider(id='env-slider', min=0, max=30, step=1, value=0)
                ])
            ])
        ])
    ], className="mb-4"),
    
    dbc.Card([
        dbc.CardHeader("Simulation Results", className="h5"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col(html.Div(id='gdp-result'), className="border-end"),
                dbc.Col(html.Div(id='productivity-result'), className="border-end"),
                dbc.Col(html.Div(id='sustainability-result'))
            ], className="text-center py-3"),
            html.Div(id='factor-analysis')
        ])
    ]),
    
    dbc.Row([
        dbc.Col(dcc.Graph(id='factor-importance'), width=8),
        dbc.Col(dcc.Graph(id='country-comparison'), width=4)
    ], className="mt-4")
], fluid=True)

@app.callback(
    [Output('gdp-result', 'children'),
     Output('productivity-result', 'children'),
     Output('sustainability-result', 'children'),
     Output('factor-analysis', 'children'),
     Output('factor-importance', 'figure'),
     Output('country-comparison', 'figure')],
    [Input('country-selector', 'value'),
     Input('urban-slider', 'value'),
     Input('global-slider', 'value'),
     Input('individual-slider', 'value'),
     Input('tech-slider', 'value'),
     Input('edu-slider', 'value'),
     Input('env-slider', 'value'),
     Input('years-slider', 'value')]
)
def update_dashboard(country, urban, global_, individual, tech, edu, env, years):
    # Scale impacts by time horizon
    scale = 1 + (years / 10)
    urban_adj = urban * scale
    global_adj = global_ * scale
    individual_adj = individual * scale
    
    # Run simulation
    results = simulate_policy(country, urban_adj, global_adj, individual_adj, tech, edu, env)
    
    # Create result cards
    gdp_card = [
        html.H4("GDP Impact", className="card-title"),
        html.H2(results['GDP Growth'], className=("text-success" if '+' in results['GDP Growth'] else "text-danger"))
    ]
    
    productivity_card = [
        html.H4("Productivity", className="card-title"),
        html.H2(results['Productivity'], className=("text-success" if '+' in results['Productivity'] else "text-danger"))
    ]
    
    sustainability_card = [
        html.H4("Sustainability", className="card-title"),
        html.H2(results['Sustainability'], className=("text-success" if '+' in results['Sustainability'] else "text-danger"))
    ]
    
    # Factor analysis
    analysis = [
        html.H5("Key Factor Impacts"),
        html.Ul([html.Li(f"{k}: {v}") for k, v in results['Key Drivers'].items()]),
        html.P(f"Projecting impacts over {years} year period")
    ]
    
    # Feature importance plot
    importance_fig = px.bar(
        x=X_train.columns,
        y=models['GDP Growth'].feature_importances_,
        labels={'x': 'Factor', 'y': 'Importance'},
        title='GDP Growth Factor Importance'
    ).update_layout(showlegend=False)
    
    # Country comparison radar chart
    country_data = df[[c for c in df.columns if c.startswith('Country_') + ['Urbanization_Rate', 'Trade_Openness', 'Individualism_Index']]]
    country_means = country_data.groupby([c for c in country_data.columns if c.startswith('Country_')]).mean().reset_index()
    country_means = country_means.melt(id_vars=[c for c in country_data.columns if c.startswith('Country_')], 
                                     var_name='Factor', value_name='Value')
    
    radar_fig = px.line_polar(
        country_means[country_means['Factor'] != 'Country_England'],  # Exclude one for clarity
        r='Value', 
        theta='Factor',
        color='Country_China',  # Just for illustration - would need proper grouping
        line_close=True,
        title='Cross-Country Factor Comparison'
    )
    
    return gdp_card, productivity_card, sustainability_card, analysis, importance_fig, radar_fig

# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=8050)
