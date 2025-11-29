import matplotlib.pyplot as plt
import matplotlib
import time
import mne
import pandas as pd
import threading

from brainaccess.utils import acquisition
from brainaccess.core.eeg_manager import EEGManager
from hotb_starter_code.FD import FractalEmotionModel

matplotlib.use("TKAgg", force=True)


class DataAcquisition:

    def __init__(self):
        self.emotion_model = FractalEmotionModel(sampling_rate=128)
        self.device_name = "BA MINI 045"
        self.eeg = acquisition.EEG()
        self.mgr = EEGManager()
        # self.mgr.connect(self.device_name)
        self.data : mne.io.Raw = None
        self.halo: dict = {
        0: "AF3",
        1: "AF4",
        2: "F3",
        3: "F4",
        4: "FC5",
        5: "FC6",
        6: "O2",
        7: "O1",
        }
        alpha = [7,14] # Alpha:   8   – 13  Hz   → Relaxed wakefulness, calm focus
        beta = [14,30] # Beta:    13  – 30  Hz   → Active thinking, alertness, problem-solving
        self.bands_freq = [alpha, beta]
        self.run = False

    def send_annotate(self):
        with EEGManager() as mgr:
            print("send_annotate")
            self.eeg.setup(mgr, device_name=self.device_name, cap=self.halo, sfreq=250)
            print("self.eeg.setup")
            # Start acquiring data
            self.run = True
            self.eeg.start_acquisition()
            print("Acquisition started")
            time.sleep(3)

            # annotation = 1
            while self.run:
                print("self.run")
                time.sleep(1)
                # send annotation to the device
                # print(f"Sending annotation {annotation} to the device")
                # self.eeg.annotate(str(annotation))
                # annotation += 1
                self.data = self.eeg.get_mne(tim=1,samples=250)
                pckg = self.process_mne()
            self.stop_recording(mgr)

    def set_run(self, new_run:bool):
        self.run = new_run

    def process_mne(self):
        self.filter_data()
        pckg = self.get_data()
        return pckg

    def stop_recording(self, mgr):    
        self.run = False
        print("Preparing to plot data")
        time.sleep(2)
        # get all eeg data and stop acquisition
        self.eeg.stop_acquisition()
        mgr.disconnect()       
        self.eeg.close()

    def filter_data(self):
        if self.data is None:
            return
        self.data.notch_filter(50,)
        print("notch")
        # 2 Hz to 42 Hz
        self.data.filter(2, 42)
        print("filter")

    def get_power_band(self, spectrum:mne.time_frequency.Spectrum, band:list):
        fmin, fmax = band
        power, freqs = spectrum.get_data(return_freqs=True, fmin=fmin, fmax=fmax)
        return power, freqs

    def extract_all_power_bands(self, spectrum:mne.time_frequency.Spectrum):
        power_bands = []
        for band in self.bands_freq:
            power_bands.append(self.get_power_band(spectrum, band))
        return power_bands

    def sum_channels(self, power) -> float:
        n = 0
        band_sum = 0
        for ch in power:
            for val in ch:
                n+=1
                band_sum += val
        print("sum")
        return band_sum / n

    def calculate_dominance(self):
        if self.data is None:
            return
        psd = self.data.compute_psd(tmin=0, tmax=60, fmin=2, fmax=50)
        bands = self.extract_all_power_bands(psd)
        alpha = bands[0]
        beta = bands[1]

        a_sum = self.sum_channels(alpha[0])
        b_sum = self.sum_channels(beta[0])
        dominance = (a_sum + b_sum) / 2
        print("dominance")
        return dominance
    
    def calculate_arousal(self):
        if self.data is None:
            return
        _, arousal= self.get_emotion()
        return arousal
        
    def get_emotion(self):
        if self.data is None:
            return 
        
        raw_af3 = self.data.copy().pick_channels(['AF3']).get_data()[0]
        raw_f4  = self.data.copy().pick_channels(['AF4']).get_data()[0]
        raw_fc6 = self.data.copy().pick_channels(['F3']).get_data()[0]
        valence,arousal =self.emotion_model.predict_window(raw_af3,raw_f4,raw_fc6)
        
        self._last_valence = valence
        self._last_arousal = arousal
        return valence,arousal
        
    def calculate_valence(self):
        if self.data is None:
            return 
        valence, _ = self.get_emotion()
        return valence
        
    def mne2pd(self):
        return self.data.to_data_frame()

    def get_data(self):
        pckg = {}
        pckg["valence"] = self.calculate_valence()
        pckg["arousal"] = self.calculate_arousal()
        pckg["dominance"] = self.calculate_dominance()
        print(pckg)
        return pckg


if __name__ == "__main__":
    data_ac = DataAcquisition()
    data_getter = threading.Thread(target=data_ac.send_annotate, daemon=True)
    data_getter.start()

    pckg = data_ac.process_mne()
    print(pckg)
    data_getter.join()