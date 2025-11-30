# emotion_image_selector.py
"""
Moduł dobierający najbliższe zdjęcie z datasetu OASIS
na podstawie znormalizowanej (0–1) walencji i pobudzenia
oraz generujący pośrednie kroki (checkpointy) w przestrzeni PAD.

Zakładamy plik `final_eeg_dataset.csv` o strukturze:
Theme,Class_Label,Valence_mean,Arousal_mean
Dog 6.jpg,Joy,6.49,5.03
...
"""

from __future__ import annotations
import math
from typing import Tuple, Sequence

import pandas as pd

# Stałe zakresów z opisu
VALENCE_MIN, VALENCE_MAX = 1.11, 6.49
AROUSAL_MIN, AROUSAL_MAX = 1.69, 5.47

DATASET_PATH = "final_eeg_dataset.csv"


def _scale_to_dataset_range(v_norm: float, a_norm: float) -> Tuple[float, float]:
    """
    Skaluje walencję i pobudzenie z zakresu [0, 1] do zakresów datasetu OASIS.
    Wartości spoza [0, 1] są przycinane (clamp).
    """
    v_norm_clamped = max(0.0, min(1.0, float(v_norm)))
    a_norm_clamped = max(0.0, min(1.0, float(a_norm)))

    v_real = VALENCE_MIN + v_norm_clamped * (VALENCE_MAX - VALENCE_MIN)
    a_real = AROUSAL_MIN + a_norm_clamped * (AROUSAL_MAX - AROUSAL_MIN)

    return v_real, a_real


def load_oasis_dataset(path: str = DATASET_PATH) -> pd.DataFrame:
    """
    Ładuje dataset OASIS do DataFrame i pilnuje, żeby były wymagane kolumny.
    """
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        # Fallback - tworzymy pusty DF, żeby kod się nie wywalił od razu
        print(f"UWAGA: Nie znaleziono pliku {path}. Tworzę przykładowy dataset.")
        return pd.DataFrame({
            "Theme": ["Astronaut 1.jpg"],
            "Valence_mean": [3.5],
            "Arousal_mean": [3.5]
        })

    required_cols = {"Theme", "Valence_mean", "Arousal_mean"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Brakuje kolumn w CSV: {missing}")

    return df


def get_closest_theme(
    v_norm: float,
    a_norm: float,
    df: pd.DataFrame | None = None,
    *,
    recent_themes: Sequence[str] | None = None,
    avoid_repeats: bool = True,
    repeat_window: int = 3,
) -> str:
    """
    Zwraca nazwę pliku z kolumny `Theme` z datasetu `final_eeg_dataset.csv`.
    """
    if df is None:
        df = load_oasis_dataset()

    v_target, a_target = _scale_to_dataset_range(v_norm, a_norm)

    distances = (
        (df["Valence_mean"] - v_target) ** 2
        + (df["Arousal_mean"] - a_target) ** 2
    ) ** 0.5

    # Jeśli nie unikamy powtórek albo nie mamy historii – klasyczne zachowanie
    if (not avoid_repeats) or not recent_themes or repeat_window <= 0:
        closest_idx = distances.idxmin()
        return df.loc[closest_idx, "Theme"]

    # bierzemy tylko ostatnie repeat_window motywów
    recent_slice = list(recent_themes)[-repeat_window:]
    recent_set = set(recent_slice)

    # sortujemy indeksy po odległości (od najbliższych)
    sorted_idx = distances.sort_values().index

    # szukamy najbliższego Theme, którego nie ma w ostatnich N wyświetleniach
    for idx in sorted_idx:
        theme = df.loc[idx, "Theme"]
        if theme not in recent_set:
            return theme

    # fallback: jeśli wszystkie są w historii, bierzemy najbliższy
    closest_idx = distances.idxmin()
    return df.loc[closest_idx, "Theme"]


def generate_path(
    tar_a: float,
    tar_v: float,
    tar_d: float,  # ignorowane, ale zostawione dla spójności interfejsu
    cur_a: float,
    cur_v: float,
    cur_d: float,  # ignorowane
    step: float,
) -> Tuple[float, float, bool]:
    """
    Generuje pojedynczy krok w stronę punktu docelowego w przestrzeni (Arousal, Valence).
    """
    if step <= 0:
        raise ValueError("Parametr 'step' musi być dodatni.")

    da = tar_a - cur_a
    dv = tar_v - cur_v
    dist = math.sqrt(da * da + dv * dv)

    if dist == 0 or dist <= step:
        next_a = tar_a
        next_v = tar_v
        reached = True
    else:
        scale = step / dist
        next_a = cur_a + da * scale
        next_v = cur_v + dv * scale
        reached = False

    next_a = max(0.0, min(1.0, next_a))
    next_v = max(0.0, min(1.0, next_v))

    return next_a, next_v, reached