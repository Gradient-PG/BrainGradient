# emotion_image_selector.py
"""
Moduł dobierający najbliższe zdjęcie z datasetu OASIS
na podstawie znormalizowanej (0–1) walencji i pobudzenia.

Zakładamy plik `final_eeg_dataset.csv` o strukturze:
Theme,Class_Label,Valence_mean,Arousal_mean
Dog 6.jpg,Joy,6.49,5.03
...
"""

from __future__ import annotations
import pandas as pd
from typing import Tuple

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
    df = pd.read_csv(path)

    required_cols = {"Theme", "Valence_mean", "Arousal_mean"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Brakuje kolumn w CSV: {missing}")

    return df


def get_closest_theme(
    v_norm: float,
    a_norm: float,
    df: pd.DataFrame | None = None,
) -> str:
    """
    Zwraca nazwę pliku z kolumny `Theme` z datasetu `final_eeg_dataset.csv`,
    którego (Valence_mean, Arousal_mean) jest najbliżej podanych
    znormalizowanych wartości (v_norm, a_norm) w [0, 1].

    Parametry:
        v_norm: walencja w zakresie [0, 1]
        a_norm: pobudzenie w zakresie [0, 1]
        df: opcjonalny już wczytany DataFrame (dla wydajności)

    Zwraca:
        Nazwę pliku (string) z kolumny `Theme`.
    """
    if df is None:
        df = load_oasis_dataset()

    v_target, a_target = _scale_to_dataset_range(v_norm, a_norm)

    # Euklidesowa odległość w przestrzeni (Valence, Arousal)
    distances = (
        (df["Valence_mean"] - v_target) ** 2
        + (df["Arousal_mean"] - a_target) ** 2
    ) ** 0.5

    closest_idx = distances.idxmin()
    return df.loc[closest_idx, "Theme"]
