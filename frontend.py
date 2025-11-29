from dash import Dash, dcc, html
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import time


import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objects as go

DECAY_COEF = 0.6
BLOB_SIZE_EXPONENT = 20


# 1. Define the Logic Functions
def create_3d_figure(df):
    fig = px.scatter_3d(
        x=df["x"],
        y=df["y"],
        z=df["z"],
        size=df["distance"],
        color=df["distance"],
        opacity=0.8,
        # Używamy Twojej palety dla punktów (opcjonalnie)
        color_continuous_scale=[[0, '#1F0034'], [0.5, '#572CD5'], [1, '#E5BB54']],
    )
    fig.update_traces(marker=dict(line=dict(width=0)))

    # KLUCZOWE: Usunięcie tła i zmiana kolorów siatki
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',  # Przezroczyste tło całego canvasu
        plot_bgcolor='rgba(0,0,0,0)',  # Przezroczyste tło wykresu
        coloraxis_showscale=False,
        scene=dict(
            xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="#572CD5", zerolinecolor="#9570FF",
                       showbackground=False, title_font=dict(color="#E5BB54")),
            yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="#572CD5", zerolinecolor="#9570FF",
                       showbackground=False, title_font=dict(color="#E5BB54")),
            zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="#572CD5", zerolinecolor="#9570FF",
                       showbackground=False, title_font=dict(color="#E5BB54")),
        ),
        margin=dict(l=0, r=0, b=0, t=0)  # Usunięcie marginesów
    )
    return fig


def create_history_areaplot(df: pd.DataFrame):
    df_melted = df.melt(
        value_vars=["arousal", "valence", "dominance"], id_vars=["step"]
    )
    df_melted = df_melted[df_melted["step"].max() == df_melted["step"]]

    # Mapa kolorów dla słupków
    color_map = {
        "arousal": "#9570FF",  # Jasny fiolet
        "valence": "#E5BB54",  # Żółty
        "dominance": "#572CD5"  # Główny fiolet
    }

    fig = px.bar(
        df_melted,
        x="variable",
        y="value",
        color="variable",
        color_discrete_map=color_map
    )

    # KLUCZOWE: Ciemny motyw dla wykresu 2D
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#E0E0E0'),  # Jasna czcionka
        showlegend=False,
        margin=dict(l=40, r=20, b=30, t=20),
        xaxis=dict(title=None, gridcolor='rgba(149, 112, 255, 0.2)'),
        yaxis=dict(title=None, gridcolor='rgba(149, 112, 255, 0.2)', zerolinecolor='#9570FF')
    )
    return fig


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


def get_clean_live_df():
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


global_df = get_clean_live_df()
global_vect = get_clean_vect()
global_target = np.array([1, 1, 0])
global_history_df = pd.DataFrame(
    data={"arousal": [0], "valence": [0], "dominance": [0], "step": [0]}
)
global_now = time.time()

app = dash.Dash(__name__)

app.layout = html.Div(
    [
        html.H1("AuraCloud - Real-time Emotion Mapping from EEG", style={"textAlign": "center"}),
        html.Div(
            [
                dcc.Graph(
                    id="live-graph",
                    figure=create_3d_figure(get_clean_live_df()),
                    style={
                        "height": "100%",
                        "width": "50%",
                    },
                ),
                dcc.Graph(
                    id="history-graph",
                    figure=create_history_areaplot(global_history_df),
                    style={
                        "height": "30%",
                        "width": "50%",
                    },
                ),
            ],
            style={
                "height": "90vh",
                "width": "90vw",
                "margin": 0,
                "padding": 0,
                "overflow": "hidden",
                "display": "flex",
                "flexDirection": "row",
            },
        ),
        dcc.Interval(id="interval-component", interval=100, n_intervals=0),
    ]
)


@app.callback(
    [
        Output("live-graph", "figure", allow_duplicate=True),
        Output("history-graph", "figure", allow_duplicate=True),
    ],
    [
        Input("live-graph", "figure"),
        Input("interval-component", "n_intervals"),
    ],
    prevent_initial_call=True,
)
def update_metrics(old_3d_fig, n_intervals):
    global global_df
    global global_vect
    global global_target
    global global_history_df
    global global_now

    ctx = callback_context
    if not ctx.triggered:
        trigger_id = "No triggers"
    else:
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    try:
        global_df.head()  # Convert JSON back to DataFrame
    except (ValueError, TypeError):
        # Fallback in case of bad data
        global_df = get_clean_live_df()

    if trigger_id == "interval-component":
        # print(
        #     "diff",
        #     global_vect - global_target,
        #     np.abs(global_vect - global_target).sum(),
        # )

        diff = time.time() - global_now
        if diff > 0.2:
            print("time diff", diff)
        global_now = time.time()



        if np.abs(global_vect - global_target).sum() < 0.3:
            global_target = np.random.random((3,))

            target_dict = {
                "arousal": global_target[0],
                "valence": global_target[1],
                "dominance": global_target[2],
                "step": len(global_history_df),
            }

            global_history_df.loc[len(global_history_df)] = target_dict
            print("new target")

        global_df = decay_data(global_df)
        global_vect = update_vector(global_vect, global_target)
        global_df = add_measurement(global_df, global_vect)

    # Create the new figure

    old_3d_fig["data"][0]["marker"]["color"] = global_df["distance"].to_numpy()
    old_3d_fig["data"][0]["marker"]["size"] = global_df["distance"].to_numpy() * 10000

    history_fig = create_history_areaplot(global_history_df)

    return old_3d_fig, history_fig


app.run(debug=True, use_reloader=True)
