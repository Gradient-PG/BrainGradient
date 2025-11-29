import numpy as np
import mne
from sklearn.ensemble import RandomForestRegressor
from scipy.integrate import simpson
from scipy.signal import welch
from scipy.signal import butter, filtfilt

def calculate_band_power(data, sf, band):
    """
    Calculate absolute power of a specific frequency band using PSD.
    """
    band_low, band_high = band
    
    # Compute Power Spectral Density (PSD) using Welch method
    win = int(4 * sf)  # 4-second window
    freqs, psd = welch(data, sf, nperseg=win)
    
    # Select PSD values within the frequency band
    idx_band = np.logical_and(freqs >= band_low, freqs <= band_high)
    
    # Integrate PSD over the band using Simpson's rule
    band_power = simpson(psd[idx_band], dx=freqs[1] - freqs[0])
    return band_power

def extract_features(raw, bands=None):
    """
    Extract band power features from raw EEG data.
    """
    if bands is None:
        bands = {'Theta': (4, 8), 'Alpha': (8, 12), 'Beta': (12, 30), 'Gamma': (30, 45)}
    
    # Epoch the data (1-second non-overlapping windows)
    epochs = mne.make_fixed_length_epochs(raw, duration=1.0, overlap=0.0)
    data = epochs.get_data()  # Shape: (n_epochs, n_channels, n_times)
    
    fs = raw.info['sfreq']
    features = []

    for epoch in data:
        # Average across channels
        avg_signal = np.mean(epoch, axis=0)
        epoch_features = [calculate_band_power(avg_signal, fs, band) for band in bands.values()]
        features.append(epoch_features)
    
    return np.array(features)

def process_and_predict_pad(raw_file_path, model_P, model_A, model_D):
    """
    Full pipeline: Raw EEG -> Features -> PAD predictions
    """
    # 1. Load raw EEG data
    raw = mne.io.read_raw_fif(raw_file_path, preload=True)
    
    # 2. Preprocessing
    raw.filter(2., 42., fir_design='firwin')          # Bandpass 1-50 Hz
    raw.set_eeg_reference('average', projection=True) # Re-reference
    
    # 3. Feature extraction
    X = extract_features(raw)
    
    # 4. Predict PAD values
    pleasure  = model_P.predict(X)
    arousal   = model_A.predict(X)
    dominance = model_D.predict(X)
    
    return list(zip(pleasure, arousal, dominance))


def higuchi_fd(time_series, k_max):
    """
    Implements the Higuchi Fractal Dimension (HFD) algorithm.
    Ref: Equations (1), (2), and (3) in the paper.
    """
    N = len(time_series)
    X = np.array(time_series)
    L_k = []
    
    # Iterate through k values from 1 to k_max
    for k in range(1, k_max + 1):
        L_m_k = []
        
        # Calculate Length L_m(k) for each m - Equation (2)
        for m in range(1, k + 1):
            # Create the decimated series - Equation (1)
            # Python is 0-indexed, so we adjust indices (m-1)
            indices = np.arange(m - 1, N, k)
            current_series = X[indices]
            n_samples = len(current_series)
            
            # Sum of absolute differences
            diffs = np.abs(np.diff(current_series))
            normalization = (N - 1) / (n_samples * k)
            L_m = (np.sum(diffs) * normalization) / k
            
            L_m_k.append(L_m)
        
        # Average L(k) over m - Equation (3) context
        L_k.append(np.mean(L_m_k))
    
    # Log-Log plot to find D (slope) - Equation (3)
    # relationship: <L(k)> ~ k^(-D)
    x = np.log(1.0 / np.arange(1, k_max + 1))
    y = np.log(L_k)
    
    # Fit linear regression to find slope
    slope, _ = np.polyfit(x, y, 1)
    
    return slope # This is the Fractal Dimension (D)

class FractalEmotionModel:
    def __init__(self, sampling_rate=128):
        self.fs = sampling_rate
        # Thresholds must be calibrated per user or set to defaults.
        # The paper suggests calibration (training) is best.
        self.arousal_threshold_high = 1.90  # Example value based on Table I logic
        self.arousal_threshold_low = 1.80   # Example value based on Table I logic
        self.valence_threshold = 0.0        # Asymmetry > 0 vs < 0
        
    def bandpass_filter(self, data, lowcut=2.0, highcut=42.0):
        """
        Applies 2-42 Hz bandpass filter as specified in[cite: 138].
        """
        nyq = 0.5 * self.fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(5, [low, high], btype='band')
        return filtfilt(b, a, data)

    def predict_window(self, raw_af3, raw_f4, raw_fc6):
        """
        Processes a single window (e.g., 1024 samples) to predict emotion.
        """
        # 1. Filter Data [cite: 211]
        filt_af3 = self.bandpass_filter(raw_af3)
        filt_f4 = self.bandpass_filter(raw_f4)
        filt_fc6 = self.bandpass_filter(raw_fc6)
        
        # 2. Calculate Higuchi FD [cite: 212]
        # k_max is usually set between 6 and 20 for EEG; 
        # The paper implies optimization but k=10 is standard for this calculation.
        fd_af3 = higuchi_fd(filt_af3, k_max=10)
        fd_f4 = higuchi_fd(filt_f4, k_max=10)
        fd_fc6 = higuchi_fd(filt_fc6, k_max=10)
        
        # 3. Determine Arousal (FC6) 
        # Paper: Higher FD = Higher Arousal [cite: 151]
        arousal_level = 0
        if fd_fc6 > self.arousal_threshold_high:
            arousal_level = 2 # High Arousal
        elif fd_fc6 > self.arousal_threshold_low:
            arousal_level = 1 # Medium Arousal
        else:
            arousal_level = 0 # Low Arousal
            
        # 4. Determine Valence (AF3 - F4) 
        # Paper tests lateralization hypothesis: Left (AF3) vs Right (F4)
        valence_score = fd_af3 - fd_f4
        valence_level = 1 if valence_score > self.valence_threshold else 0
        
        
        return valence_level,arousal_level


mapping = {
            (0, 0): "Sad",
            (0, 1): "Frustrated",
            (0, 2): "Fear",
            (1, 0): "Satisfied",
            (1, 1): "Pleasant",
            (1, 2): "Happy"
        }
pad_model = FractalEmotionModel(sampling_rate=128)

raw = mne.io.read_raw_fif("data/2.fif", preload=True)

raw_af3 = raw.copy().pick_channels(['AF3']).get_data()[0]
raw_f4  = raw.copy().pick_channels(['AF4']).get_data()[0]
raw_fc6 = raw.copy().pick_channels(['F3']).get_data()[0]

# Run prediction
emotion = pad_model.predict_window(raw_af3, raw_f4, raw_fc6)

print(f"Predicted Emotion: {emotion}")