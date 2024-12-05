from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import kagglehub
from plotly.colors import sample_colorscale
import os
from seaborn import color_palette
import StatstoStars.src.prediction as prediction
import StatstoStars.src.player_analysis as player_analysis
import StatstoStars.src.about as about
from StatstoStars.src.prediction import create_first_graph


color_palette = px.colors.qualitative.Dark24


# Initialize Dash app
app = Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.LITERA])
app.title = "Stats to Stars - NFL Analytics"
server = app.server

# Download dataset using kagglehub
try:
    path = kagglehub.dataset_download("philiphyde1/nfl-stats-1999-2022")
    week_df = pd.DataFrame()
    year_df = pd.DataFrame()
    for file_name in os.listdir(path):
        file_path = os.path.join(path, file_name)
        if "weekly_player_data" in file_name:
            week_df = pd.read_csv(file_path)
        elif "yearly_player_data" in file_name:
            year_df = pd.read_csv(file_path)
except Exception as e:
    print(f"Error loading datasets: {e}")

# Data Preprocessing
year_df.fillna({'reception_td': 0, 'run_td': 0, 'receptions': 0, 'receiving_yards': 0, 'rushing_yards': 0}, inplace=True)
year_df['total_touchdowns'] = year_df['reception_td'] + year_df['run_td']
year_df['total_yardage'] = year_df['receiving_yards'] + year_df['rushing_yards']
week_df['Total Touchdowns'] = week_df['reception_td'] + week_df['run_td']
week_df['Total Yards'] = week_df['receiving_yards'] + week_df['rushing_yards']

# New player data
touchdown_df = pd.DataFrame({
    "Player": [
        "Austin Ekeler", "Christian McCaffrey", "Davante Adams",
        "Derrick Henry", "Ezekiel Elliot", "Jalen Hurts",
        "Jonathan Taylor", "Mike Evans", "Tyreek Hill"
    ],
    "Total Touchdowns": [42.0, 22.0, 45.0, 41.0, 34.0, 26.0, 37.0, 33.0, 35.0],
})

reception_df = pd.DataFrame({
    "Player": [
        "Ceedee Lamb", "Cooper Kupp", "Davante Adams",
        "Justin Jefferson", "Keenan Allen", "Michael Pittman",
        "Stefon Diggs", "Travis Kelce", "Tyreek Hill", "Amon-ra St.Brown"
    ],
    "Total Receptions": [260, 312, 338, 324, 272, 227, 338, 307, 317, 196],
})

yards_df = pd.DataFrame({
    "Player": [
        "Aaron Jones", "Austin Ekeler", "Christian McCaffrey",
        "Davante Adams", "Derrick Henry", "Jonathan Taylor",
        "Justin Jefferson", "Nick Chubb", "Tyreek Hill"
    ],
    "Total Yardage": [5056.0, 5170.0, 5055.0, 5591.0, 6553.0, 5537.0, 5943.0, 4605.0, 6290.0],
})

# Filter the DataFrame for the 2023 season
players_2023_df = week_df[week_df['season'] == 2023]

# Calculate the total touchdowns, receptions, and yardage for each player
# Use .loc to explicitly modify columns in the DataFrame
players_2023_df = players_2023_df.copy()
players_2023_df.loc[:, 'total_touchdowns'] = players_2023_df['reception_td'] + players_2023_df['run_td']
players_2023_df.loc[:, 'total_yards'] = players_2023_df['receiving_yards'] + players_2023_df['rushing_yards']


# Group by player and calculate the averages
average_stats_2023 = (
    players_2023_df.groupby('player_name')
    .agg({
        'total_touchdowns': 'mean',  # Average weekly total touchdowns
        'receptions': 'mean',        # Average weekly receptions
        'total_yards': 'mean'        # Average weekly total yards
    })
    .rename(columns={
        'total_touchdowns': 'Avg Total Touchdowns',
        'receptions': 'Avg Receptions',
        'total_yards': 'Avg Total Yards'
    })
)

num_touchdown_players = len(touchdown_df)
num_reception_players = len(reception_df)
num_yardage_players = len(yards_df)

num_players = 3  # Number of players
pinkyl_colors = sample_colorscale('Pinkyl', [i / (num_players - 1) for i in range(num_players)])

touchdown_colors = sample_colorscale('Pinkyl', [i / (num_touchdown_players - 1) for i in range(num_touchdown_players)])
reception_colors = sample_colorscale('Pinkyl', [i / (num_reception_players - 1) for i in range(num_reception_players)])
yardage_colors = sample_colorscale('Pinkyl', [i / (num_yardage_players - 1) for i in range(num_yardage_players)])

top_3_touchdowns = average_stats_2023['Avg Total Touchdowns'].nlargest(3)
top_3_receptions = average_stats_2023['Avg Receptions'].nlargest(3)
top_3_yards = average_stats_2023['Avg Total Yards'].nlargest(3)
top_3_fig = make_subplots(
    rows=1, cols=3, 
    subplot_titles=[
        'Most Average Touchdowns', 
        'Most Average Receptions', 
        'Most Average Yardage'
    ],
    column_widths=[0.3, 0.3, 0.3]
)


# Add Top 3 Touchdowns
top_3_fig.add_trace(
    go.Bar(
        x=top_3_touchdowns.index,
        y=top_3_touchdowns.values,
        name='Average Total Touchdowns',
        marker=dict(color=px.colors.sequential.Sunsetdark[0])  # Apply a color
    ),
    row=1, col=1
)

# Add Top 3 Receptions
top_3_fig.add_trace(
    go.Bar(
        x=top_3_receptions.index,
        y=top_3_receptions.values,
        name='Average Receptions',
        marker=dict(color=px.colors.sequential.Sunsetdark[1])  # Apply a color
    ),
    row=1, col=2
)

# Add Top 3 Yardage
top_3_fig.add_trace(
    go.Bar(
        x=top_3_yards.index,
        y=top_3_yards.values,
        name='Average Total Yards',
        marker=dict(color=px.colors.sequential.Sunsetdark[2])  # Apply a color
    ),
    row=1, col=3
)

# Update the layout of the figure
top_3_fig.update_layout(
    title='Top 3 Players Comparison for 2023 Season',
    showlegend=False,
    template='plotly_white',
    height=500,
    bargap=0.3
)

# Normalize seasons to start at t=0 for each player
def normalize_seasons(player_data):
    player_data = player_data.sort_values('season')  # Ensure sorted by season
    player_data['t'] = range(len(player_data))  # Create t=0, t=1, etc.
    return player_data

normalized_df = (
    year_df.groupby('player_name', group_keys=False, as_index=False)
    .apply(normalize_seasons)
    .reset_index(drop=True)
)

def create_wr_baseline_comparison(normalized_df, selected_players, player_colors):
    # Filter for WR and calculate the baseline
    wr_df = normalized_df[normalized_df['position'].str.upper() == 'WR']  # Ensure 'WR' matches correctly

    # Calculate total_touchdowns and total_yards for baseline
    wr_df['total_touchdowns'] = wr_df['reception_td'] + wr_df['run_td']
    wr_df['total_yards'] = wr_df['receiving_yards'] + wr_df['rushing_yards']
    wr_df['total_performance'] = (
        wr_df['total_touchdowns'] + wr_df['total_yards'] + wr_df['receptions']
    )

    # Group by t and calculate the baseline (mean values for each time step)
    baseline_df = wr_df.groupby('t').agg(total_performance=('total_performance', 'mean')).reset_index()

    # Calculate total performance for selected players
    selected_df = normalized_df[normalized_df['player_name'].isin(selected_players)]

    selected_df['total_touchdowns'] = selected_df['reception_td'] + selected_df['run_td']
    selected_df['total_yards'] = selected_df['receiving_yards'] + selected_df['rushing_yards']
    selected_df['total_performance'] = (
        selected_df['total_touchdowns'] + selected_df['total_yards'] + selected_df['receptions']
    )

    # Merge selected players with the baseline data on 't'
    comparison_df = selected_df.merge(baseline_df, on='t', suffixes=('_player', '_baseline'))

    # Create a Plotly figure
    fig = go.Figure()

    # Plot baseline total performance
    fig.add_trace(go.Scatter(
        x=baseline_df['t'],
        y=baseline_df['total_performance'],
        mode='lines',
        name='Baseline Total Performance',
        line=dict(color='gray', dash='dash'),
        legendgroup='baseline'
    ))

    # Plot each player's performance
    for player_name in selected_players:
        player_data = comparison_df[comparison_df['player_name'] == player_name]
        fig.add_trace(go.Scatter(
            x=player_data['t'],
            y=player_data['total_performance_player'],
            mode='lines+markers',
            name=f"Total Performance: {player_name}",
            line=dict(color=player_colors[player_name]),
            marker=dict(size=8)
        ))

    # Update layout for the figure
    fig.update_layout(
        title="Comparison of Selected Players Against WR Baseline",
        xaxis_title="Normalized Time (t)",
        yaxis_title="Performance Metrics",
        template='plotly_white',
        legend_title='Players',
        hovermode='closest',
        height=600
    )

    return fig



# Sample data
players = ["Justin Jefferson", "Davante Adams", "Tyreek Hill"]
rf_predictions = [85.3, 90.7, 78.2]
seasons_played = [4, 10, 8]


# Call the function to generate the graph
wr_baseline_graph = create_wr_baseline_comparison(normalized_df)
# Calculate weighted scores
max_seasons = max(seasons_played)
weights = [max_seasons / sp for sp in seasons_played]
weighted_rf_predictions = [rf * weight for rf, weight in zip(rf_predictions, weights)]

# Navbar
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dcc.Link("Home", href="/", className="nav-link")),
        dbc.NavItem(dcc.Link("Prediction", href="/prediction", className="nav-link")),
        dbc.NavItem(dcc.Link("Player Analysis", href="/player-analysis", className="nav-link")),
        dbc.NavItem(dcc.Link("About", href="/about", className="nav-link")),
    ],
    brand="Stats to Stars - NFL Analytics",
    color="black",
    dark=True,
)

# About Page Layout

about_layout = html.Div(
    [
        html.H1("About Stats to Stars", style={"textAlign": "center", "marginBottom": "20px"}),
        html.Div(
            [
                html.H3("Welcome to Stats to Stars"),
                html.P(
                    "Stats to Stars is an interactive platform that provides in-depth insights and "
                    "visualizations of NFL player performance. Designed for fans, analysts, coaches, and "
                    "sponsors, the app allows users to explore, analyze, and compare the performance of top "
                    "players across recent seasons."
                ),
                html.H3("Purpose"),
                html.P(
                    "The app aims to bridge the gap between raw data and actionable insights by offering visually "
                    "compelling dashboards and analytics. It empowers users to:"
                ),
                html.Ul(
                    [
                        html.Li("Analyze player performance trends across seasons."),
                        html.Li("Identify top players based on key metrics like touchdowns, receptions, and yardage."),
                        html.Li("Compare players' career and weekly performance to make informed decisions."),
                        html.Li("Explore interactive charts to uncover hidden patterns and insights."),
                    ]
                ),
                html.H3("Key Features"),
                html.Ul(
                    [
                        html.Li("Seasonal Analysis: Explore the top performers for each season (2012-2023) in categories like touchdowns, receptions, and total yardage."),
                        html.Li("Weekly Performance: Dive into detailed weekly stats for the 2023 season, including averages and top 3 players for touchdowns, receptions, and yardage."),
                        html.Li("Career Comparisons: Compare career stats of standout players, including their average touchdowns, receptions, and games per season."),
                        html.Li("Interactive Graphs: Intuitive and interactive visualizations for seamless exploration of player data."),
                        html.Li("Consistent Performers: Identify players who consistently rank in the top 10 across multiple seasons."),
                    ]
                ),
                html.H3("Who Is This For?"),
                html.Ul(
                    [
                        html.Li("Fans: Discover how your favorite players stack up against the competition."),
                        html.Li("Coaches and Analysts: Use insights to strategize and refine gameplay."),
                        html.Li("Sponsors: Find the best players to endorse by evaluating their consistent performance and marketability."),
                        html.Li("Fantasy Football Enthusiasts: Make informed picks and trades for your fantasy team."),
                    ]
                ),
                html.H3("Data Sources"),
                html.P(
                    "All data used in this app is sourced from reliable platforms, including the NFL and Kaggle repositories, ensuring accuracy and up-to-date information."
                ),
                html.H3("Acknowledgments"),
                html.P(
                    "This app was developed as part of an educational and exploratory project, with contributions from "
                    "Salomé Rivas and inspired by the passion for sports analytics and data visualization."
                ),
            ],
            style={"lineHeight": "1.6", "fontSize": "16px", "margin": "0 50px"},
        ),
    ],
    style={"padding": "20px", "backgroundColor": "#f9f9f9"},
)

# Player Analysis Layout
player_layout = html.Div([
   html.H1("NFL Player Performance Analysis", style={"textAlign": "center", "marginBottom": "20px"}),

    # Slider for selecting seasons
    html.Div([
        html.Label("Select Seasons:"),
        dcc.RangeSlider(
            id="season-slider",
            min=year_df['season'].min(),
            max=year_df['season'].max(),
            step=1,
            marks={year: str(year) for year in range(year_df['season'].min(), year_df['season'].max() + 1)},
            value=[2020, 2023],  # Default range
            tooltip={"placement": "bottom", "always_visible": True}
        )
    ], style={"margin": "20px"}),

    # Graph container for averages
    dcc.Graph(id="averages-graph", style={"marginTop": "20px"}),

    # Graph containers for individual player metrics
    dcc.Graph(id="touchdowns-graph", style={"marginTop": "20px"}),
    dcc.Graph(id="yardage-graph", style={"marginTop": "20px"}),
    dcc.Graph(id="receptions-graph", style={"marginTop": "20px"}),

    # Player Comparison Section
    html.H1("Player Comparison with Dynamic Season Selection", style={"textAlign": "center", "marginTop": "40px"}),

    # Dropdowns for player selection
    html.Div([
        html.Div([
            html.Label("Select Player 1:"),
            dcc.Dropdown(id="player1-dropdown", style={"width": "130%"})
        ], style={"display": "inline-block", "margin": "10px"}),

        html.Div([
            html.Label("Select Player 2:"),
            dcc.Dropdown(id="player2-dropdown", style={"width": "130%"})
        ], style={"display": "inline-block", "margin": "10px"}),

        html.Div([
            html.Label("Select Player 3:"),
            dcc.Dropdown(id="player3-dropdown", style={"width": "130%"})
        ], style={"display": "inline-block", "margin": "10px"})
    ]),

    # Graph to display the comparison
    dcc.Graph(id="comparison-graph", style={"marginTop": "20px"}),

    html.H1("Player Comparison with Baseline", style={"textAlign": "center", "marginBottom": "20px"}),

    # Dropdown for selecting players
    html.Div([
        html.Label("Select Players for Comparison:"),
        dcc.Dropdown(
            id="players-dropdown",
            options=[],  # Options will be populated dynamically
            multi=True,
            value=[],
            placeholder="Select up to 3 players",
            style={"width": "60%", "margin": "0 auto"}
        )
    ], style={"textAlign": "center", "marginBottom": "20px"}),

    # Graph to display the baseline comparison
    dcc.Graph(id="baseline-comparison-graph", style={"marginTop": "20px"}),
])


# Prediction Layout
prediction_layout = html.Div([
    # Prediction Dashboard Section
    html.Div([
        html.H1("Prediction Dashboard", style={"textAlign": "center", "marginBottom": "30px"}),
        html.H3("2020-2023 Yardage, Touchdown, and Reception Analysis", style={"textAlign": "center"}),
        html.P("Average per Metric", style={"textAlign": "center", "fontSize": "16px"}),
     

         # Add the first graph
    html.Div(
    dcc.Graph(
        id="first-graph",
        figure=create_first_graph(),
        style={"marginTop": "20px"}
    ),
    style={
        "display": "flex",
        "justifyContent": "center",
        "alignItems": "center"
    }
),

   html.H3("Top 10 Players per Metric", style={"textAlign": "center", "fontSize": "2px"}),


        dcc.Graph(
            figure=go.Figure(
                data=[
                    go.Bar(
                        x=touchdown_df["Player"],
                        y=touchdown_df["Total Touchdowns"],
                        marker=dict(color=touchdown_colors),  # Use custom colors
                        name="Total Touchdowns"
                    )
                ],
                layout=go.Layout(
                    title="Total Touchdowns",
                    xaxis=dict(title="Player"),
                    yaxis=dict(title="Total Touchdowns"),
                    template="plotly_white"
                )
            )
        )
    ], style={"padding": "20px"}),

    # Receptions Graph Section
    html.Div([
        dcc.Graph(
            figure=go.Figure(
                data=[
                    go.Bar(
                        x=reception_df["Player"],
                        y=reception_df["Total Receptions"],
                        marker=dict(color=reception_colors),  # Use custom colors
                        name="Total Receptions"
                    )
                ],
                layout=go.Layout(
                    title="Total Receptions",
                    xaxis=dict(title="Player"),
                    yaxis=dict(title="Total Receptions"),
                    template="plotly_white"
                )
            )
        )
    ], style={"padding": "20px"}),

    # Yardage Graph Section
    html.Div([
        dcc.Graph(
            figure=go.Figure(
                data=[
                    go.Bar(
                        x=yards_df["Player"],
                        y=yards_df["Total Yardage"],
                        marker=dict(color=yardage_colors),  # Use custom colors
                        name="Total Yardage"
                    )
                ],
                layout=go.Layout(
                    title="Total Yardage",
                    xaxis=dict(title="Player"),
                    yaxis=dict(title="Total Yardage"),
                    template="plotly_white"
                )
            )
        )
    ], style={"padding": "20px"}),


    # Weekly Data Analysis Section
    html.Div([
        html.H1("2023 Weekly Player Analysis", style={"textAlign": "center", "marginBottom": "30px"}),
        html.H3("Average Weekly Performance Metrics", style={"textAlign": "center"}),
        html.P(
            "This section showcases the top 3 players in average touchdowns, receptions,"
            "and yardage for the 2023 season. The latest season plays a crucial role in evaluating football players,"
            "as their careers often evolve rapidly. By focusing on the most recent data, this analysis provides valuable "
            "insights to inform predictions and strategies for the upcoming 2024 season.",
            style={"textAlign": "center", "fontSize": "16px"}
        ),
        html.Div([
        dcc.Graph(figure=top_3_fig)
    ], style={"padding": "20px"}),

    html.P(
        "Justin Jefferson stands out as the only wide receiver (WR) to rank among the top three players in more than one category across the three graphs.",
        style={"textAlign": "center", "fontSize": "16px", "color": "grey"}
    ),

    # Players Excelling in Multiple Metrics Section
html.Div([
    html.H3("Players Excelling in Multiple Metrics", style={"textAlign": "center"}),
    html.P("These players have appeared in multiple top categories over the last 4 seasons", style={"textAlign": "center"}),
    html.Ul([
        html.Li("Justin Jefferson"),
        html.Li("Davante Adams"),
        html.Li("Tyreek Hill"),
    ], style={"textAlign": "center", "fontSize": "16px", "listStyleType": "none", "padding": "0"}),
], style={"padding": "20px"}),


    # Side-by-Side Graphs for Selected Players
    html.Div([
        html.H1("Player Performance Analysis", style={"textAlign": "center", "marginBottom": "30px"}),
        html.H3("Justin Jefferson vs. Davante Adams vs. Tyreek Hill", style={"textAlign": "center"}),
        html.Div([
    html.Div(dcc.Graph(
        figure=go.Figure(
            data=[
                go.Bar(
                    x=["Justin Jefferson", "Davante Adams", "Tyreek Hill"],
                    y=[8, 10, 4],
                    marker=dict(color=pinkyl_colors),  # Apply Pinkyl colors
                    name="Avg Receptions"
                )
            ],
            layout=go.Layout(
                title="Average Receptions per Season",
                xaxis=dict(
                    title="Player",
                    tickangle=0  # Move tickangle here
                ),
                yaxis=dict(title="Avg Receptions"),
                template="plotly_white"
            )
        )
    ), style={"width": "40%", "display": "inline-block", "padding": "5px"}),

    html.Div(dcc.Graph(
        figure=go.Figure(
            data=[
                go.Bar(
                    x=["Justin Jefferson", "Davante Adams", "Tyreek Hill"],
                    y=[30, 96, 85],
                    marker=dict(color=pinkyl_colors),  # Apply Pinkyl colors
                    name="Avg Touchdowns"
                )
            ],
            layout=go.Layout(
                title="Average Touchdowns per Season",
                xaxis=dict(
                    title="Player",
                    tickangle=0  # Move tickangle here
                ),
                yaxis=dict(title="Avg Touchdowns"),
                template="plotly_white"
            )
        )
    ), style={"width": "40%", "display": "inline-block", "padding": "5px"}),

    html.Div(dcc.Graph(
        figure=go.Figure(
            data=[
                go.Bar(
                    x=["Justin Jefferson", "Davante Adams", "Tyreek Hill"],
                    y=[60, 155, 136],
                    marker=dict(color=pinkyl_colors),  # Apply Pinkyl colors
                    name="Avg Games"
                )
            ],
            layout=go.Layout(
                title="Average Games per Season",
                xaxis=dict(
                    title="Player",
                    tickangle=0  # Move tickangle here
                ),
                yaxis=dict(title="Avg Games"),
                template="plotly_white"
            )
        )
    ), style={"width": "40%", "display": "inline-block", "padding": "5px"})
], style={"display": "flex", "justifyContent": "center", "alignItems": "center"}),

# New WR Baseline Comparison Section
    html.Div([
        html.H1("WR Baseline Comparison", style={"textAlign": "center", "marginBottom": "20px"}),
    dcc.Graph(
        id="wr-baseline-comparison",
        figure=wr_baseline_graph,
        style={"padding": "20px"}
    )
    ]),

    ]),
html.H1("Machine Learning Results: Weighted Random Forest Scores", style={"textAlign": "center", "marginTop": "40px"}),

html.Div([
    dcc.Graph(
        id="weighted-scores-graph",
        figure=go.Figure(
            data=[
                go.Bar(
                    x=players,  # Player names
                    y=weighted_rf_predictions,  # Weighted scores from the updated calculation
                    name="Weighted Random Forest Scores",
                    marker_color="green"  # Removed text and textposition
                )
            ],
            layout=go.Layout(
                title="Weighted Sponsorship Scores by Player",
                xaxis_title="Player",
                yaxis_title="Weighted Sponsorship Score",
                template="plotly_white",
                legend_title="Score Type"
            )
        )
    )
], style={"padding": "20px"}),


    # Sponsorship Recommendation Section
    html.Div([
        html.H3("Player Sponsorship Recommendation", style={"textAlign": "center", "marginTop": "35px"}),
        html.H1("Justin Jefferson", style={"textAlign": "center", "color": "purple", "fontSize": "50px"}),
        html.P(
            "Justin Jefferson’s exceptional yardage and receptions performance make him the ideal"
            "candidate to be Nike's standout athlete for 2024.",
            style={"textAlign": "center", "fontSize": "20px"}
        ),
    ], style={"padding": "20px", "backgroundColor": "#f5f5f5", "borderRadius": "10px"})
])
])

app.layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),
        navbar,
        html.Div(id="page-content", style={"padding": "20px"}),
    ]
)

@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def display_page(pathname):
    if pathname == "/prediction":
        return prediction_layout  # No parentheses
    elif pathname == "/player-analysis":
        return player_layout  # No parentheses
    elif pathname == "/about":
        return about_layout  # No parentheses
    else:
        return html.Div(
            [
                html.H1("Welcome to Stats to Stars", style={"textAlign": "center", "marginBottom": "20px"}),
                html.P(
                    "Analyze NFL player performance and uncover insights for sponsorship opportunities.",
                    style={"textAlign": "center", "fontSize": "18px"},
                ),
                html.Div(
                    dbc.Button("Get Started", href="/player-analysis", color="primary", size="lg"),
                    style={"textAlign": "center", "marginTop": "30px"},
                ),
            ]
        )
    
    

@app.callback(
    [
        Output("touchdowns-graph", "figure"),
        Output("yardage-graph", "figure"),
        Output("receptions-graph", "figure"),
        Output("averages-graph", "figure"),
    ],
    [Input("season-slider", "value")]
)
def update_all_graphs(season_range):
    # Generate bar graphs for touchdowns, yardage, and receptions
    touchdowns_data = player_analysis.calculate_total_touchdowns(year_df, season_range)
    yardage_data = player_analysis.calculate_total_yardage(year_df, season_range)
    receptions_data = player_analysis.calculate_total_receptions(year_df, season_range)

    touchdowns_fig = player_analysis.generate_bar_graph(
        touchdowns_data,
        title="Top 10 Touchdown Leaders (Selected Seasons)",
        x_label="Player",
        y_label="Touchdowns"
    )

    yardage_fig = player_analysis.generate_bar_graph(
        yardage_data,
        title="Top 10 Yardage Leaders (Selected Seasons)",
        x_label="Player",
        y_label="Yardage"
    )

    receptions_fig = player_analysis.generate_bar_graph(
        receptions_data,
        title="Top 10 Receptions Leaders (Selected Seasons)",
        x_label="Player",
        y_label="Receptions"
    )

    # Generate the averages graph
    averages_fig = player_analysis.generate_average_graphs(season_range)

    return touchdowns_fig, yardage_fig, receptions_fig, averages_fig

@app.callback(
    [Output("player1-dropdown", "options"),
     Output("player2-dropdown", "options"),
     Output("player3-dropdown", "options")],
    [Input("season-slider", "value")]
)
def update_player_dropdowns_callback(season_range):
    return player_analysis.update_player_dropdowns(season_range)

@app.callback(
    Output("comparison-graph", "figure"),
    [
        Input("season-slider", "value"),
        Input("player1-dropdown", "value"),
        Input("player2-dropdown", "value"),
        Input("player3-dropdown", "value")
    ]
)
def update_comparison_graph(season_range, player1, player2, player3):
    return player_analysis.update_graph(season_range, player1, player2, player3)


@app.callback(
    [Output("players-dropdown", "options"), 
     Output("baseline-comparison-graph", "figure")],
    [Input("players-dropdown", "value")]
)
def update_dropdown_and_graph(selected_players):
    # Populate dropdown options with unique player names
    options = [{"label": player, "value": player} for player in sorted(normalized_df['player_name'].unique())]

    # If no players are selected, return default options and an empty graph
    if not selected_players or len(selected_players) > 3:
        return options, go.Figure()

    # Assign dynamic colors to the selected players
    player_colors = {
        player: color_palette[i % len(color_palette)]
        for i, player in enumerate(selected_players)
    }

    # Create and return the baseline comparison graph
    baseline_graph = create_wr_baseline_comparison(normalized_df, selected_players, player_colors)
    return options, baseline_graph


# Run Server
if __name__ == "__main__":
    app.run_server(debug=True)
