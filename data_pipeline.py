import matplotlib.pyplot as plt
import matplotlib
import time
import mne
import pandas as pd
import threading

from brainaccess.utils import acquisition
from brainaccess.core.eeg_manager import EEGManager

matplotlib.use("TKAgg", force=True)


class DataAcquisition:

    def __init__(self):
        self.device_name = "BA MINI 045"
        self.eeg = acquisition.EEG()
        # self.mgr = EEGManager()
        # self.mgr.connect(self.device_name)
        self.data : mne.io.Raw = None
        self.halo: dict = {
        0: "F4",
        1: "F3",
        2: "C4",
        3: "C3",
        4: "P4",
        5: "P3",
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

            annotation = 1
            while self.run:
                print("self.run")
                time.sleep(1)
                # send annotation to the device
                # print(f"Sending annotation {annotation} to the device")
                # self.eeg.annotate(str(annotation))
                # annotation += 1
                self.data = self.eeg.get_mne(tim=1,samples=250)
                pckg = self.process_mne()


    def process_mne(self):
        self.filter_data()
        pckg = self.get_data()
        return pckg

    def stop_recording(self):    
        self.run = False
        print("Preparing to plot data")
        time.sleep(2)
        # get all eeg data and stop acquisition
        self.eeg.get_mne()
        self.eeg.stop_acquisition()
        # self.mgr.disconnect()       
        # self.data = self.eeg.data.mne_raw
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

    def mne2pd(self):
        return self.data.to_data_frame()

    def get_data(self):
        pckg = {}
        pckg["Valence"] = None
        pckg["Arousal"] = None
        pckg["Dominance"] = self.calculate_dominance()
        print(pckg)
        return pckg


if __name__ == "__main__":
    data_ac = DataAcquisition()
    data_getter = threading.Thread(target=data_ac.send_annotate, daemon=True)
    data_getter.start()

    pckg = data_ac.process_mne()
    print(pckg)
    data_getter.join()

# # Access data as NumPy arrays
# data, times = mne_raw.get_data(return_times=True)
# print(f"Data shape: {data.shape}")

# # save EEG data to MNE fif format
# eeg.data.save(f'./data/{time.strftime("%Y%m%d_%H%M")}-raw.fif')
# # Close brainaccess library
# eeg.close()
# # conversion to microvolts
# mne_raw.apply_function(lambda x: x*10**-6)
# # Show recorded data
# mne_raw.filter(1, 40).plot(scalings="auto", verbose=False)
# plt.show()