import os
import urllib.parse
from dash import Dash, dcc, html, Input, Output, State, callback_context
from flask import send_from_directory
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import time
from data_pipeline import DataAcquisition

# --- KONFIGURACJA (BEZ ZMIAN) ---
DECAY_COEF = 0.6
BLOB_SIZE_EXPONENT = 20

COLORS = {
    "background": "#1F0034",
    "primary": "#572CD5",
    "secondary": "#9570FF",
    "accent": "#E5BB54",
    "text": "#E0E0E0",
}

# --- FUNKCJE LOGIKI ---
# (Identyczne jak w Twoim kodzie bazowym, tylko z poprawionymi kolorami wykresów)


def create_3d_figure(df):
    fig = px.scatter_3d(
        df,
        x="x",
        y="y",
        z="z",
        size="distance",
        color="distance",
        opacity=0.8,
        # color_continuous_scale=[
        #     [0, COLORS["background"]],
        #     [0.5, COLORS["primary"]],
        #     [1, COLORS["accent"]],
        # ],
    )
    fig.update_traces(marker=dict(line=dict(width=0)))
    fig.update_traces(
        marker=dict(symbol="diamond"),
        # selector=dict(mode="`markers"),
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",  # Przezroczyste tło całego canvasu
        plot_bgcolor="rgba(0,0,0,0)",  # Przezroczyste tło wykresu
        coloraxis_showscale=False,
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            #             "arousal"
            # "valence"
            # "dominance"
            xaxis=dict(
                title=dict(text="Arousal"),
                backgroundcolor="rgba(0,0,0,0)",
                gridcolor=COLORS["primary"],
                showbackground=False,
                title_font=dict(color=COLORS["accent"], size=40),
            ),
            yaxis=dict(
                title=dict(text="Valence"),
                backgroundcolor="rgba(0,0,0,0)",
                gridcolor=COLORS["primary"],
                showbackground=False,
                title_font=dict(color=COLORS["accent"], size=40),
            ),
            zaxis=dict(
                title=dict(text="Dominance"),
                backgroundcolor="rgba(0,0,0,0)",
                gridcolor=COLORS["primary"],
                showbackground=False,
                title_font=dict(color=COLORS["accent"], size=40),
            ),
        ),
    )

    df_reference = pd.DataFrame(
        {
            "name": [
                "Safe",
                "Satisfied",
                "Surprised",
                # "Happy",
                "Sad",
                "Unbothered",
                "Scared",
                "Angry",
            ],
            "valence": [0.75, 0.75, 0.75, 0.75, 0.25, 0.25, 0.25],
            "arousal": [0.25, 0.25, 0.75, 0.75, 0.25, 0.75, 0.75],
            "dominance": [0.25, 0.75, 0.25, 0.75, 0.75, 0.25, 0.75],
        }
    )

    df_reference["valence"] += 0.25
    df_reference["arousal"] += 0.25
    df_reference["dominance"] += 0.25

    fig.add_trace(
        go.Scatter3d(
            x=df_reference["arousal"],
            y=df_reference["valence"],
            z=df_reference["dominance"],
            mode="markers+text",
            text=df_reference["name"],
            textposition="top center",
            textfont=dict(color="yellow", size=20),
            marker=dict(
                size=6,
                color="white",
                opacity=1.0,
                line=dict(width=2, color="black"),
                symbol="circle",
            ),
            showlegend=False,
        )
    )
    return fig


def create_barplot_figure(data_dict: dict):
    # Mapa kolorów dla słupków
    color_map = {
        "arousal": "#9570FF",  # Jasny fiolet
        "valence": "#E5BB54",  # Żółty
        "dominance": "#572CD5",  # Główny fiolet
    }

    fig = px.bar(
        x=data_dict.keys(),
        y=data_dict.values(),
        color=data_dict.keys(),
        color_discrete_map=color_map,
    )

    # KLUCZOWE: Ciemny motyw dla wykresu 2D
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E0E0E0"),  # Jasna czcionka
        showlegend=False,
        margin=dict(l=40, r=20, b=30, t=20),
        xaxis=dict(title=None, gridcolor="rgba(149, 112, 255, 0.2)"),
        yaxis=dict(
            title=None, gridcolor="rgba(149, 112, 255, 0.2)", zerolinecolor="#9570FF"
        ),
    )
    return fig


def decay_data(df):
    df["distance"] = df["distance"] * DECAY_COEF
    df.loc[df["distance"] < 0.1, "distance"] = 0
    return df


def update_vector(base_vector, update_vector):
    return base_vector * 0.9 + update_vector * 0.1


def add_measurement(df, vect):
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


# --- STAN GLOBALNY ---
global_df = get_clean_live_df()
global_vect = get_clean_vect()
global_target = np.array([1, 1, 0])
global_now = time.time()
global_data_aquisition = DataAcquisition()
global_target_history = []

# --- APP SETUP ---
app = Dash(__name__)
server = app.server

# Ścieżka do obrazka
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATH = "NewDataset/Astronaut 1.jpg"
FULL_PATH = os.path.join(BASE_DIR, IMAGE_PATH)

if os.path.exists(FULL_PATH):
    IMG_SRC = "/NewDataset/Astronaut 1.jpg"
else:
    IMG_SRC = "https://images.unsplash.com/photo-1446776811953-b23d57bd21aa?w=500&auto=format&fit=crop"


@server.route("/NewDataset/<path:path>")
def serve_dataset(path):
    return send_from_directory(os.path.join(BASE_DIR, "NewDataset"), path)


# --- STYL MODALA ---
MODAL_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "width": "100vw",
    "height": "100vh",
    "backgroundColor": "rgba(31, 0, 52, 0.95)",  # Ciemny fiolet, lekko przeźroczysty
    "zIndex": 9999,
    "display": "none",  # Domyślnie ukryty
    "justifyContent": "center",
    "alignItems": "center",
    "flexDirection": "column",
    "backdropFilter": "blur(10px)",
}

app.layout = html.Div(
    style={
        "backgroundColor": COLORS["background"],
        "minHeight": "100vh",
        "width": "100vw",
        "overflow": "hidden",
        "fontFamily": "sans-serif",
        "color": COLORS["text"],
        "display": "flex",
        "flexDirection": "column",
    },
    children=[
        # --- MECHANIZM PROGRAMOWEGO ODPALANIA ---
        dcc.Store(id="modal-store", data={"open": False}),
        # 1. MODAL (Warstwa wierzchnia)
        html.Div(
            id="modal-container",
            style=MODAL_STYLE,
            children=[
                html.Div(
                    id="modal-overlay",
                    style={
                        "position": "absolute",
                        "top": 0,
                        "left": 0,
                        "width": "100%",
                        "height": "100%",
                        "cursor": "pointer",
                    },
                ),
                html.Img(
                    id="modal-image",  # ID potrzebne do zamykania po kliknięciu
                    src=IMG_SRC,
                    # ZMIANA: Wymuszamy duży rozmiar (height/width zamiast max-)
                    style={
                        "height": "85vh",  # Wymuszona wysokość
                        "width": "85vw",  # Wymuszona szerokość
                        "minHeight": "600px",  # Minimalna wysokość (żeby nie było za małe)
                        "minWidth": "800px",  # Minimalna szerokość
                        "objectFit": "contain",  # Zachowaj proporcje, ale wypełnij ramkę
                        "zIndex": 10000,
                        "position": "relative",
                        "border": f"2px solid {COLORS['accent']}",
                        "borderRadius": "12px",
                        "boxShadow": "0 0 40px rgba(229, 187, 84, 0.3)",
                        "cursor": "pointer",
                        "backgroundColor": "rgba(0,0,0,0.5)",  # Ciemne tło pod samym zdjęciem, jeśli proporcje są inne
                    },
                ),
                html.Div(
                    "Kliknij zdjęcie lub tło, aby zamknąć",
                    style={
                        "marginTop": "15px",
                        "color": COLORS["secondary"],
                        "zIndex": 10001,
                        "pointerEvents": "none",
                        "fontSize": "14px",
                    },
                ),
            ],
        ),
        # 2. HEADER
        html.Div(
            [
                html.H1(
                    "AuraCloud",
                    style={
                        "color": COLORS["accent"],
                        "margin": 0,
                        "textAlign": "center",
                        "paddingTop": "10px",
                    },
                ),
                html.Div(
                    "Real-time Emotion Mapping",
                    style={
                        "textAlign": "center",
                        "color": COLORS["secondary"],
                        "marginBottom": "10px",
                    },
                ),
                # Przycisk DEMO
                html.Button(
                    "Test Fullscreen Function",
                    id="btn-trigger-modal",
                    style={
                        "background": "transparent",
                        "border": f"1px solid {COLORS['secondary']}",
                        "color": COLORS["secondary"],
                        "padding": "5px 10px",
                        "borderRadius": "5px",
                        "cursor": "pointer",
                        "position": "absolute",
                        "top": "20px",
                        "right": "20px",
                    },
                ),
            ],
            style={"position": "relative"},
        ),
        # 3. GŁÓWNY OBSZAR
        html.Div(
            style={
                "display": "flex",
                "flex": 1,
                "padding": "20px",
                "gap": "20px",
                "height": "85vh",
                "boxSizing": "border-box",
            },
            children=[
                # LEWA KOLUMNA
                html.Div(
                    className="glass-panel",
                    style={"flex": 2, "position": "relative"},
                    children=[
                        dcc.Graph(
                            id="live-graph",
                            figure=create_3d_figure(get_clean_live_df()),
                            style={"width": "100%", "height": "95%"},
                            config={"responsive": True},
                        )
                    ],
                ),
                # PRAWA KOLUMNA
                html.Div(
                    style={
                        "flex": 1,
                        "display": "flex",
                        "flexDirection": "column",
                        "gap": "20px",
                    },
                    children=[
                        # Prawa Góra
                        html.Div(
                            className="glass-panel",
                            style={"flex": 1, "position": "relative"},
                            children=[
                                dcc.Graph(
                                    id="history-graph",
                                    figure=create_barplot_figure(
                                        {
                                            key: value
                                            for key, value in zip(
                                                ["arousal", "valence", "dominance"],
                                                global_target,
                                            )
                                        }
                                    ),
                                    style={"width": "95%", "height": "100%"},
                                    config={"responsive": True},
                                )
                            ],
                        ),
                        # Prawa Dół (Miniatura)
                        html.Div(
                            id="image-trigger",
                            className="glass-panel zoom-hover",
                            style={
                                "flex": 1,
                                "position": "relative",
                                "display": "flex",
                                "justifyContent": "center",
                                "alignItems": "center",
                                "overflow": "hidden",
                                "cursor": "pointer",
                            },
                            children=[
                                html.Img(
                                    src=IMG_SRC,
                                    style={
                                        "maxWidth": "90%",
                                        "maxHeight": "90%",
                                        "objectFit": "contain",
                                    },
                                ),
                                html.Div(
                                    "⤢",
                                    style={
                                        "position": "absolute",
                                        "bottom": "10px",
                                        "right": "10px",
                                        "color": COLORS["accent"],
                                        "fontSize": "24px",
                                    },
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
        dcc.Interval(id="interval-component", interval=100, n_intervals=0),
    ],
)

# --- CALLBACKS ---


@app.callback(
    [
        Output("live-graph", "figure", allow_duplicate=True),
        Output("history-graph", "figure", allow_duplicate=True),
    ],
    [
        Input("live-graph", "figure"),
        Input("history-graph", "figure"),
        Input("interval-component", "n_intervals"),
    ],
    prevent_initial_call=True,
)
def update_metrics(old_3d_fig, barplot_fig, n_intervals):
    global global_df
    global global_vect
    global global_target
    global global_now
    global global_data_aquisition
    global global_target_history

    ctx = callback_context
    trigger_id = (
        ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else "No triggers"
    )

    try:
        global_df.head()
    except (ValueError, TypeError, AttributeError):
        global_df = get_clean_live_df()

    if trigger_id == "interval-component":
        diff = time.time() - global_now
        if diff > 0.2:
            print("time diff", diff)
        global_now = time.time()


        prev_len = len(global_target_history)
        global_data_aquisition.data_consumer(global_target_history)
        if prev_len != len(global_target_history):
            global_target = global_target_history[-1]

        # # Random targeting
        # if np.abs(global_vect - global_target).sum() < 0.3:
        #     global_target = np.random.random((3,))

            target_dict = {
                "arousal": global_target[0],
                "valence": global_target[1],
                "dominance": global_target[2],
            }
            barplot_fig = create_barplot_figure(target_dict)

            print("new target")

        global_df = decay_data(global_df)
        global_vect = update_vector(global_vect, global_target)
        global_df = add_measurement(global_df, global_vect)

    # Create the new figure

    old_3d_fig["data"][0]["marker"]["color"] = global_df["distance"].to_numpy()
    old_3d_fig["data"][0]["marker"]["size"] = global_df["distance"].to_numpy() * 10000

    return old_3d_fig, barplot_fig


# --- CALLBACK MODALA ---
@app.callback(
    [Output("modal-container", "style"), Output("modal-store", "data")],
    [
        Input("image-trigger", "n_clicks"),
        Input("modal-overlay", "n_clicks"),
        Input("modal-image", "n_clicks"),  # DODANE: Kliknięcie w zdjęcie też zamyka
        Input("btn-trigger-modal", "n_clicks"),
        Input("modal-store", "data"),
    ],
    [State("modal-container", "style")],
    prevent_initial_call=True,
)
def toggle_modal_display(
    img_trigger_click,
    overlay_click,
    modal_img_click,
    btn_click,
    store_data,
    current_style,
):
    ctx = callback_context
    if not ctx.triggered:
        return current_style, store_data

    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    new_style = current_style.copy()
    new_store_data = store_data.copy() if store_data else {"open": False}

    # Otwieranie (Klik w miniaturę lub przycisk)
    if trigger_id == "image-trigger" or trigger_id == "btn-trigger-modal":
        new_style["display"] = "flex"
        new_store_data["open"] = True

    # Zamykanie (Klik w tło LUB klik w samo zdjęcie)
    elif trigger_id == "modal-overlay" or trigger_id == "modal-image":
        new_style["display"] = "none"
        new_store_data["open"] = False

    # Reakcja na zewnętrzną zmianę Store
    elif trigger_id == "modal-store":
        if store_data.get("open"):
            new_style["display"] = "flex"
        else:
            new_style["display"] = "none"

    return new_style, new_store_data

if __name__ == "__main__":
    # global_data_aquisition.
    app.run(debug=True, use_reloader=True)
