from dash import Dash, dcc, html
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd


import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objects as go

DECAY_COEF = 0.8
BLOB_SIZE_EXPONENT = 20


# 1. Define the Logic Functions
def create_figure(df):
    fig = px.scatter_3d(
        x=df["x"],
        y=df["y"],
        z=df["z"],
        size=df["distance"],
        color=df["distance"],
        opacity=0.7,
        color_continuous_midpoint=0.5,
    )
    fig.update_traces(marker=dict(line=dict(width=0)))
    return fig


def boost_data(df):
    vect = np.array([0.21, 0.32, 0.17])

    df["distance"] = np.pow(
        1 - (np.sqrt(np.pow(df[["x", "y", "z"]] - vect, 2).sum(axis=1)) / np.sqrt(3)),
        BLOB_SIZE_EXPONENT,
    )
    df.loc[df["distance"] < 0.1, "distance"] = 0
    return df


def decay_data(df):
    df["distance"] = df["distance"] * DECAY_COEF
    df.loc[df["distance"] < 0.1, "distance"] = 0

    return df


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
        dcc.Interval(id="interval-component", interval=100, n_intervals=0),
    ]
)

current_df = get_clean_df()


@app.callback(
    Output("live-graph", "figure"),
    [Input("interval-component", "n_intervals"), Input("boost-btn", "n_clicks")],
)
def update_metrics(n_intervals, n_clicks):
    global current_df

    ctx = callback_context
    if not ctx.triggered:
        trigger_id = "No triggers"
    else:
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    try:
        current_df.head()  # Convert JSON back to DataFrame
    except (ValueError, TypeError):
        # Fallback in case of bad data
        current_df = get_clean_df()

    # Apply Logic
    if trigger_id == "boost-btn":
        current_df = boost_data(current_df)
    elif trigger_id == "interval-component":
        current_df = decay_data(current_df)

    # Create the new figure
    fig = create_figure(current_df)

    return fig


app.run(debug=True, use_reloader=True)
