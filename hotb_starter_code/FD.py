import numpy as np
import mne
from sklearn.ensemble import RandomForestRegressor
from scipy.integrate import simpson
from scipy.signal import welch
from scipy.signal import butter, filtfilt


def higuchi_fd(time_series, k_max):
    N = len(time_series)  # number of samples
    X = np.array(time_series)  # time series
    L_k = []  # mean of a subsequence

    for k in range(1, k_max + 1):
        L_m_k = []  # length of a subsequence

        # Calculate Length L_m(k) for each m
        for m in range(1, k + 1):
            # Create the decimated series
            indices = np.arange(m - 1, N, k)
            current_series = X[indices]
            n_samples = len(current_series)

            # Sum of absolute differences
            diffs = np.abs(np.diff(current_series))
            normalization = (N - 1) / (n_samples * k)
            L_m = (np.sum(diffs) * normalization) / k

            L_m_k.append(L_m)

        L_k.append(np.mean(L_m_k))

    # Log-Log plot to find D (slope) -
    # relationship: <L(k)> ~ k^(-D)
    x = np.log(1.0 / np.arange(1, k_max + 1))
    y = np.log(L_k)

    # Fit linear regression to find slope
    slope, _ = np.polyfit(x, y, 1)
    return slope  # This is the Fractal Dimension FD


class FractalEmotionModel:
    def __init__(self, sampling_rate=128):
        self.fs = sampling_rate
        # Thresholds must be calibrated per user or set to defaults.
        # The paper suggests calibration (training) is best.
        self.arousal_threshold_high = 1.90  # Example value
        self.arousal_threshold_low = 1.80  # Example value
        self.valence_threshold = 0.0  # Asymmetry > 0 vs < 0

    def bandpass_filter(self, data, lowcut=2.0, highcut=42.0):
        nyq = 0.5 * self.fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(5, [low, high], btype="band")
        return filtfilt(b, a, data)

    def predict_window(self, raw_af3, raw_f4, raw_fc6):
        # 1. Filter Data
        filt_af3 = self.bandpass_filter(raw_af3)
        filt_f4 = self.bandpass_filter(raw_f4)
        filt_fc6 = self.bandpass_filter(raw_fc6)

        # 2. Calculate Higuchi FD
        fd_af3 = higuchi_fd(filt_af3, k_max=10)
        fd_f4 = higuchi_fd(filt_f4, k_max=10)
        fd_fc6 = higuchi_fd(filt_fc6, k_max=10)

        # 3. Determine Arousal (FC6)
        # Higher FD = Higher Arousal
        arousal_level = np.clip((fd_fc6 - 1.7) * 10, 0, 2) / 2

        # 4. Determine Valence (AF3 - F4)
        valence_score = fd_af3 - fd_f4
        valence_level = np.clip(valence_score * 1000, 0, 1)

        return valence_level, arousal_level
