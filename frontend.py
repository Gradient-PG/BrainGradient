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


# def boost_data(df):
#     vect = np.array([0.21, 0.32, 0.17])

#     df["distance"] = np.pow(
#         1 - (np.sqrt(np.pow(df[["x", "y", "z"]] - vect, 2).sum(axis=1)) / np.sqrt(3)),
#         BLOB_SIZE_EXPONENT,
#     )
#     df.loc[df["distance"] < 0.1, "distance"] = 0
#     return df


def decay_data(df):
    df["distance"] = df["distance"] * DECAY_COEF
    df.loc[df["distance"] < 0.1, "distance"] = 0

    return df


def update_vector(base_vector, update_vector):
    return base_vector * 0.9 + update_vector * 0.1


def add_measurement(df, vect):
    # vect = np.array([0.21, 0.32, 0.17])

    df["distance"] = df["distance"] + np.pow(
        1 - (np.sqrt(np.pow(df[["x", "y", "z"]] - vect, 2).sum(axis=1)) / np.sqrt(3)),
        BLOB_SIZE_EXPONENT,
    )
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


def get_clean_vect():
    return np.zeros((3,))


app = dash.Dash(__name__)

app.layout = html.Div(
    [
        html.H1("Nie w sumie nie mam żadnego pomysłu", style={"textAlign": "center"}),
        dcc.Graph(id="live-graph"),
        dcc.Interval(id="interval-component", interval=100, n_intervals=0),
    ],
    style={
        "height": "85vh",
        "width": "50vw",
        "margin": 0,
        "padding": 0,
        "overflow": "hidden",
        "alighn": "left",
    },
)

global_df = get_clean_df()
global_vect = get_clean_vect()
global_target = np.array([1, 1, 0])


@app.callback(
    Output("live-graph", "figure"),
    Input("interval-component", "n_intervals"),
)
def update_metrics(n_intervals):
    global global_df
    global global_vect
    global global_target

    ctx = callback_context
    if not ctx.triggered:
        trigger_id = "No triggers"
    else:
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    try:
        global_df.head()  # Convert JSON back to DataFrame
    except (ValueError, TypeError):
        # Fallback in case of bad data
        global_df = get_clean_df()

    # Apply Logic
    # if trigger_id == "boost-btn":
    #     current_df = boost_data(current_df)
    if trigger_id == "interval-component":
        if (global_vect - global_target).sum() < 0.1:
            global_target = np.random.random((3,))
            print(f"new target {global_target}")

        global_df = decay_data(global_df)
        global_vect = update_vector(global_vect, global_target)
        global_df = add_measurement(global_df, global_vect)
        print(global_vect)

    # Create the new figure
    fig = create_figure(global_df)

    return fig


app.run(debug=True, use_reloader=True)
