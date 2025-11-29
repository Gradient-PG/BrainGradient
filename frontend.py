from dash import Dash, dcc, html
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd


import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objects as go

DECAY_COEF = 0.8


# 1. Define the Logic Functions
def create_figure(df):
    fig = px.scatter_3d(
        x=df["x"],
        y=df["y"],
        z=df["z"],
        size=df["distance"],
        color=df["distance"],
        opacity=0.7,
    )
    fig.update_traces(marker=dict(line=dict(width=0)))
    return fig


def boost_data(df):
    vect = np.array([0.21, 0.32, 0.17])

    df["distance"] = np.pow(
        1 - (np.sqrt(np.pow(df[["x", "y", "z"]] - vect, 2).sum(axis=1)) / np.sqrt(3)),
        10,
    )
    df.loc[df["distance"] < 0.1, "distance"] = 0
    return df


def decay_data(df):
    """Decays value (floored at 0)."""

    df["distance"] = df["distance"] * DECAY_COEF
    df.loc[df["distance"] < 0.1, "distance"] = 0

    return df  # Decays by 10%


def get_clean_df():
    df = (
        pd.DataFrame(pd.Series(np.arange(11)).rename("x"))
        .merge(pd.Series(np.arange(11)).rename("y"), how="cross")
        .merge(pd.Series(np.arange(11)).rename("z"), how="cross")
    )
    df = df / 10
    df["distance"] = 0
    return df


app = dash.Dash(__name__)

app.layout = html.Div(
    [
        html.H1("Real-time Decay & Boost", style={"textAlign": "center"}),
        dcc.Graph(id="live-graph"),
        html.Div(
            [
                html.Button(
                    "update",
                    id="boost-btn",
                    n_clicks=0,
                    style={
                        "fontSize": "20px",
                        "padding": "15px 30px",
                        "backgroundColor": "#2ecc71",
                        "color": "white",
                        "border": "none",
                        "cursor": "pointer",
                    },
                )
            ],
            style={"textAlign": "center", "marginTop": "20px"},
        ),
        dcc.Interval(id="interval-component", interval=500, n_intervals=0),
        # dcc.Store(id="data-store", data={"df_base": df_base}),
        html.Div(
            id="current-value-storage", children="50.0", style={"display": "none"}
        ),
    ]
)

current_df = get_clean_df()


# 4. The Callback
@app.callback(
    [Output("live-graph", "figure"), Output("current-value-storage", "children")],
    [Input("interval-component", "n_intervals"), Input("boost-btn", "n_clicks")],
    [State("current-value-storage", "children")],
)
def update_metrics(n_intervals, n_clicks, data):
    global current_df  # Declare current_df as global

    # Determine what triggered the callback
    ctx = callback_context
    if not ctx.triggered:
        trigger_id = "No triggers"
    else:
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    try:
        current_df = pd.read_json(data)  # Convert JSON back to DataFrame
    except (ValueError, TypeError):
        # Fallback in case of bad data
        current_df = get_clean_df()

    # Apply Logic
    if trigger_id == "boost-btn":
        # Apply Boost
        current_df = boost_data(current_df)
    elif trigger_id == "interval-component":
        # Apply Decay
        current_df = decay_data(current_df)

    # Create the new figure
    fig = create_figure(current_df)

    # Return the figure and the new state as JSON
    return fig, current_df.to_json()


app.run(debug=True, use_reloader=True)
