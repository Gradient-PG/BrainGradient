import matplotlib.pyplot as plt
import matplotlib
import time
import mne
import pandas as pd
import threading
import queue

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
        alpha = [8,13] 
        beta = [13,30] 
        theta = [4, 8]
        self.bands_freq = [alpha, beta, theta]
        self.run = False
        self.data_queue = queue.Queue()

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

            while self.run:
                print("self.run")
                time.sleep(1)
                self.data = self.eeg.get_mne(tim=1,samples=250)
                pckg = self.process_mne()
                if pckg:
                    self.data_queue.put(pckg)
            self.stop_recording(mgr)

    def data_consumer(self, pckg_list):
        """ CONSUMER THREAD: Reads data from the Queue """
        print("Consumer thread started...")
        while self.run or not self.data_queue.empty():
            try:
                # 3. Get data from queue (waits up to 1 second for data)
                pckg = self.data_queue.get(timeout=1) 
                pckg_list.append(pckg)
                print(f" [Consumer Thread] Received: {pckg}")

                self.data_queue.task_done()
            except queue.Empty:
                # No data received within timeout, loop again to check self.run
                continue
            except Exception as e:
                print(f"Consumer error: {e}")

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

    def avg_channels(self, power) -> float:
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
        theta = bands[2]

        a_avg = self.avg_channels(alpha[0])
        b_avg = self.avg_channels(beta[0])
        t_avg = self.avg_channels(theta[0])
        xa = a_avg/b_avg
        xb = t_avg/b_avg
        stress = 0.5*(xa + xb)
        dominance = -self.calculate_valence() + stress
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
    
    # packages
    pckg_list = []

    data_producer = threading.Thread(target=data_ac.send_annotate, daemon=True)
    data_consumer = threading.Thread(target=data_ac.data_consumer, args=(pckg_list,), daemon=True)
    
    data_producer.start()
    data_consumer.start()

    data_consumer.join()
    data_producer.join()