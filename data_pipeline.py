import matplotlib.pyplot as plt
import matplotlib
import time
import mne
import pandas as pd

from brainaccess.utils import acquisition
from brainaccess.core.eeg_manager import EEGManager

matplotlib.use("TKAgg", force=True)




class DataAcquisition:

    def __init__(self):
        self.device_name = "BA MINI 045"
        self.eeg = acquisition.EEG()
        self.mgr = EEGManager()
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
        self.run = False

    def send_annotate(self):
        self.eeg.setup(self.mgr, device_name=self.device_name, cap=self.halo, sfreq=250)

        # Start acquiring data
        self.run = True
        self.eeg.start_acquisition()
        print("Acquisition started")
        time.sleep(3)

        annotation = 1
        while self.run:
            time.sleep(1)
            # send annotation to the device
            print(f"Sending annotation {annotation} to the device")
            self.eeg.annotate(str(annotation))
            annotation += 1

    def stop_recording(self) -> mne.raw.io:    
        self.run = False
        print("Preparing to plot data")
        time.sleep(2)
        # get all eeg data and stop acquisition
        self.eeg.get_mne()
        self.eeg.stop_acquisition()
        self.mgr.disconnect()       
        self.data = self.eeg.data.mne_raw
        self.eeg.close()

    def filter_data(self):
        self.data.notch_filter(50,)
        # 2 Hz to 42 Hz
        self.data.filter(2, 42)

    def mne2pd(self):
        return self.data.to_data_frame()


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