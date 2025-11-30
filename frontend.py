import os
import urllib.parse
from dash import Dash, dcc, html, Input, Output, State, callback_context, no_update
from flask import send_from_directory
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import time
from data_pipeline import DataAcquisition
import threading

# --- IMPORTY LOGIKI ŚCIEŻKI ---
from emotion_image_selector import load_oasis_dataset, get_closest_theme, generate_path

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

# --- ŁADOWANIE DATASETU ---
OASIS_DF = load_oasis_dataset()

# --- FUNKCJE LOGIKI ---

def create_3d_figure(df):
    fig = px.scatter_3d(
        df,
        x="x",
        y="y",
        z="z",
        size="distance",
        color="distance",
        opacity=0.8,
    )
    fig.update_traces(marker=dict(line=dict(width=0)))
    fig.update_traces(
        marker=dict(symbol="diamond"),
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False,
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis=dict(
                title=dict(text="Arousal"),
                backgroundcolor="rgba(0,0,0,0)",
                gridcolor=COLORS["primary"],
                showbackground=False,
                title_font=dict(color=COLORS["accent"], size=40),
                range=[0, 1],
            ),
            yaxis=dict(
                title=dict(text="Valence"),
                backgroundcolor="rgba(0,0,0,0)",
                gridcolor=COLORS["primary"],
                showbackground=False,
                title_font=dict(color=COLORS["accent"], size=40),
                range=[0, 1],
            ),
            zaxis=dict(
                title=dict(text="Dominance"),
                backgroundcolor="rgba(0,0,0,0)",
                gridcolor=COLORS["primary"],
                showbackground=False,
                title_font=dict(color=COLORS["accent"], size=40),
                range=[0, 1],
            ),
        ),
    )

    # POPRAWIONE WSPÓŁRZĘDNE REFERENCYJNE
    # Rozsunięto punkty, aby napisy nie nachodziły na siebie.
    # Naprawiono pozycję "Sad" (niska walencja).
    df_reference = pd.DataFrame(
        {
            "name": [
                "Safe", "Satisfied", "Surprised", "Sad", "Unbothered", "Scared", "Angry",
            ],
            "valence": [0.75, 0.75, 0.75, 0.75, 0.25, 0.25, 0.25],
            "arousal": [0.25, 0.25, 0.75, 0.75, 0.25, 0.75, 0.75],
            "dominance": [0.25, 0.75, 0.25, 0.75, 0.75, 0.25, 0.75],
        }
    )

    # Przesuwamy referencje, aby były bardziej widoczne (+0.25 przesuwa środek ciężkości)
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
    color_map = {
        "arousal": "#9570FF",
        "valence": "#E5BB54",
        "dominance": "#572CD5",
    }

    fig = px.bar(
        x=data_dict.keys(),
        y=data_dict.values(),
        color=data_dict.keys(),
        color_discrete_map=color_map,
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E0E0E0"),
        showlegend=False,
        # ZWIĘKSZONO MARGINES DOLNY (b) z 10 na 40, aby etykiety się mieściły
        margin=dict(l=40, r=20, b=40, t=10),
        xaxis=dict(title=None, gridcolor="rgba(149, 112, 255, 0.2)"),
        yaxis=dict(
            title=None, gridcolor="rgba(149, 112, 255, 0.2)", zerolinecolor="#9570FF", range=[0, 1.1]
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
global_target = np.array([0.5, 0.5, 0.5])
global_now = time.time()
global_data_aquisition = DataAcquisition()
global_target_history = []
global_recent_themes = []

# --- APP SETUP ---
app = Dash(__name__)
server = app.server

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATH = "NewDataset/Astronaut 1.jpg"
FULL_PATH = os.path.join(BASE_DIR, IMAGE_PATH)

if os.path.exists(FULL_PATH):
    DEFAULT_IMG = "/NewDataset/Astronaut 1.jpg"
else:
    DEFAULT_IMG = "https://images.unsplash.com/photo-1446776811953-b23d57bd21aa?w=500&auto=format&fit=crop"


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
    "backgroundColor": "rgba(31, 0, 52, 0.95)",
    "zIndex": 9999,
    "display": "none",
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
        dcc.Store(id="modal-store", data={"open": False}),

        # --- PATH LOGIC STORES ---
        dcc.Store(id="path-state", data={
            "is_walking": False,
            "cur_a": 0.5, "cur_v": 0.5, "cur_d": 0.5,
            "tar_a": 0.5, "tar_v": 0.5, "tar_d": 0.5,
            "step_size": 0.05
        }),
        dcc.Interval(id="path-ticker", interval=2500, disabled=True),

        # 1. MODAL
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
                    id="modal-image",
                    src=DEFAULT_IMG,
                    style={
                        "height": "85vh",
                        "width": "85vw",
                        "minHeight": "600px",
                        "minWidth": "800px",
                        "objectFit": "contain",
                        "zIndex": 10000,
                        "position": "relative",
                        "border": f"2px solid {COLORS['accent']}",
                        "borderRadius": "12px",
                        "boxShadow": "0 0 40px rgba(229, 187, 84, 0.3)",
                        "cursor": "pointer",
                        "backgroundColor": "rgba(0,0,0,0.5)",
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
                        "gap": "5px", # ZMNIEJSZONE DO 5px
                    },
                    children=[
                        # Prawa Góra (Wykres + Slidery + Guzik)
                        html.Div(
                            className="glass-panel",
                            style={
                                "flex": 1,
                                "position": "relative",
                                "display": "flex",
                                "flexDirection": "column",
                                "padding": "10px",
                                "gap": "15px" # DODANO ODSTĘP między wykresem a suwakami
                            },
                            children=[
                                # WYKRES (Góra)
                                html.Div(
                                    style={"flex": "0 0 50%", "minHeight": "0"},
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
                                            style={"width": "95%", "height": "100%", "margin": "0 auto"},
                                            config={"responsive": True},
                                        )
                                    ]
                                ),
                                # KONTROLKI (Dół)
                                html.Div(
                                    style={
                                        "flex": 1,
                                        "padding": "5px 20px 5px 20px", # Zmniejszone paddingi
                                        "display": "flex",
                                        "flexDirection": "column",
                                        "justifyContent": "center",
                                        "gap": "5px" # Zmniejszony gap między suwakami a guzikiem
                                    },
                                    children=[
                                        html.Div([
                                            html.Label("Target Arousal (A)", style={"color": COLORS["accent"], "fontWeight": "bold", "fontSize": "11px"}),
                                            dcc.Slider(
                                                id="slider-arousal",
                                                min=0, max=1, step=0.05, value=0.5,
                                                marks={0: '0', 0.5: '0.5', 1: '1'},
                                                tooltip={"placement": "bottom", "always_visible": True}
                                            )
                                        ]),
                                        html.Div([
                                            html.Label("Target Valence (V)", style={"color": COLORS["accent"], "fontWeight": "bold", "fontSize": "11px"}),
                                            dcc.Slider(
                                                id="slider-valence",
                                                min=0, max=1, step=0.05, value=0.5,
                                                marks={0: '0', 0.5: '0.5', 1: '1'},
                                                tooltip={"placement": "bottom", "always_visible": True}
                                            )
                                        ]),
                                        # NOWY TRZECI SLIDER
                                        html.Div([
                                            html.Label("Target Dominance (D)", style={"color": COLORS["accent"], "fontWeight": "bold", "fontSize": "11px"}),
                                            dcc.Slider(
                                                id="slider-dominance",
                                                min=0, max=1, step=0.05, value=0.5,
                                                marks={0: '0', 0.5: '0.5', 1: '1'},
                                                tooltip={"placement": "bottom", "always_visible": True}
                                            )
                                        ]),
                                        # GUZIK
                                        html.Button(
                                            "Generate Target Emotion Path",
                                            id="btn-start-path",
                                            style={
                                                "background": COLORS["primary"],
                                                "color": "white",
                                                "border": "none",
                                                "padding": "8px",
                                                "borderRadius": "5px",
                                                "cursor": "pointer",
                                                "fontWeight": "bold",
                                                "marginTop": "0px",
                                                "width": "100%",
                                                "fontSize": "13px"
                                            }
                                        )
                                    ]
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
                                    id="thumbnail-img",
                                    src=DEFAULT_IMG,
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
        Output("path-state", "data"),
        Output("path-ticker", "disabled"),
        Output("thumbnail-img", "src"),
        Output("modal-image", "src"),
    ],
    [
        Input("btn-start-path", "n_clicks"),
        Input("path-ticker", "n_intervals")
    ],
    [
        State("path-state", "data"),
        State("slider-arousal", "value"),
        State("slider-valence", "value"),
        State("slider-dominance", "value"),
        State("thumbnail-img", "src")
    ],
    prevent_initial_call=True
)
def manage_path(btn_click, n_intervals, state, slider_a, slider_v, slider_d, current_img):
    ctx = callback_context
    if not ctx.triggered:
        return no_update

    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    # 1. Start ścieżki
    if trigger_id == "btn-start-path":
        cur_a = state.get("cur_a", 0.5)
        cur_v = state.get("cur_v", 0.5)
        cur_d = state.get("cur_d", 0.5)

        print(f"START PATH: ({cur_a:.2f}, {cur_v:.2f}) -> CEL: ({slider_a:.2f}, {slider_v:.2f})")

        new_state = state.copy()
        new_state.update({
            "is_walking": True,
            "tar_a": slider_a,
            "tar_v": slider_v,
            "tar_d": slider_d
        })
        return new_state, False, current_img, current_img

    # 2. Krok ścieżki
    elif trigger_id == "path-ticker":
        if not state["is_walking"]:
            return no_update, True, no_update, no_update

        next_a, next_v, reached = generate_path(
            tar_a=state["tar_a"],
            tar_v=state["tar_v"],
            tar_d=state["tar_d"],
            cur_a=state["cur_a"],
            cur_v=state["cur_v"],
            cur_d=state["cur_d"],
            step=state["step_size"]
        )

        # Interpolacja dominacji
        tar_d = state["tar_d"]
        cur_d = state["cur_d"]
        diff_d = tar_d - cur_d
        if abs(diff_d) < state["step_size"]:
            next_d = tar_d
        else:
            next_d = cur_d + np.sign(diff_d) * state["step_size"]

        theme = get_closest_theme(
            v_norm=next_v,
            a_norm=next_a,
            df=OASIS_DF,
            recent_themes=global_recent_themes,
            avoid_repeats=True,
            repeat_window=3
        )

        global_recent_themes.append(theme)
        if len(global_recent_themes) > 10:
            global_recent_themes.pop(0)

        new_img_src = f"/NewDataset/{theme}"

        new_state = state.copy()
        new_state.update({
            "cur_a": next_a,
            "cur_v": next_v,
            "cur_d": next_d,
            "is_walking": not reached
        })

        return new_state, reached, new_img_src, new_img_src

    return no_update


@app.callback(
    [
        Output("live-graph", "figure", allow_duplicate=True),
        Output("history-graph", "figure", allow_duplicate=True),
    ],
    [
        Input("interval-component", "n_intervals"),
        Input("path-state", "data")
    ],
    [
        State("live-graph", "figure"),
        State("history-graph", "figure"),
        State("slider-arousal", "value"),
        State("slider-valence", "value"),
        State("slider-dominance", "value")
    ],
    prevent_initial_call=True,
)
def update_metrics(n_intervals, path_state, old_3d_fig, barplot_fig, s_a, s_v, s_d):
    global global_df
    global global_vect
    global global_target
    global global_now
    global global_data_aquisition
    global global_target_history

    ctx = callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else "No triggers"

    # Bezpieczeństwo - gdyby grafy nie istniały
    if old_3d_fig is None:
        old_3d_fig = create_3d_figure(get_clean_live_df())
    if barplot_fig is None:
        barplot_fig = create_barplot_figure({"arousal":0.5, "valence":0.5, "dominance":0.5})

    try:
        global_df.head()
    except (ValueError, TypeError, AttributeError):
        global_df = get_clean_live_df()

    # --- GLÓWNA LOGIKA AKTUALIZACJI CELU ---

    # 1. TRYB ŚCIEŻKI: Jeśli idziemy -> cel to aktualny krok
    if path_state and path_state.get("is_walking", False):
        target_a = path_state["cur_a"]
        target_v = path_state["cur_v"]
        target_d = path_state["cur_d"]
        global_target = np.array([target_a, target_v, target_d])

        target_dict = {
            "arousal": global_target[0],
            "valence": global_target[1],
            "dominance": global_target[2],
        }
        barplot_fig = create_barplot_figure(target_dict)

    # 2. TRYB SPOCZYNKU: Zamiast random/EEG -> cel to pozycja sliderów
    elif trigger_id == "interval-component":
        prev_len = len(global_target_history)
        global_data_aquisition.data_consumer(global_target_history)

        # TUTAJ ZMIANA: Zamiast random, bierzemy ze sliderów (State s_a, s_v, s_d)
        global_target = np.array([s_a, s_v, s_d])

        target_dict = {
            "arousal": global_target[0],
            "valence": global_target[1],
            "dominance": global_target[2],
        }
        # Aktualizujemy barplot, żeby pokazywał cel (slidery)
        barplot_fig = create_barplot_figure(target_dict)

    # --- WSPÓLNA FIZYKA CHMURY ---
    if trigger_id in ["interval-component", "path-state"]:
        global_df = decay_data(global_df)
        global_vect = update_vector(global_vect, global_target)
        global_df = add_measurement(global_df, global_vect)

        old_3d_fig["data"][0]["marker"]["color"] = global_df["distance"].to_numpy()
        old_3d_fig["data"][0]["marker"]["size"] = global_df["distance"].to_numpy() * 10000

    return old_3d_fig, barplot_fig


# --- CALLBACK MODALA (BEZ ZMIAN) ---
@app.callback(
    [Output("modal-container", "style"), Output("modal-store", "data")],
    [
        Input("image-trigger", "n_clicks"),
        Input("modal-overlay", "n_clicks"),
        Input("modal-image", "n_clicks"),
        Input("btn-trigger-modal", "n_clicks"),
        Input("modal-store", "data"),
    ],
    [State("modal-container", "style")],
    prevent_initial_call=True,
)
def toggle_modal_display(img_clk, overlay_clk, modal_clk, btn_clk, store_data, style):
    ctx = callback_context
    if not ctx.triggered:
        return style, store_data

    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    new_style = style.copy()
    new_store = store_data.copy() if store_data else {"open": False}

    if trigger_id in ["image-trigger", "btn-trigger-modal"]:
        new_style["display"] = "flex"
        new_store["open"] = True
    elif trigger_id in ["modal-overlay", "modal-image"]:
        new_style["display"] = "none"
        new_store["open"] = False
    elif trigger_id == "modal-store":
        new_style["display"] = "flex" if new_store.get("open") else "none"

    return new_style, new_store

if __name__ == "__main__":
    pckg_list = []
    app.run(debug=True, use_reloader=True)