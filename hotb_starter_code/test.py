""" EEG measurement example

Example how to get measurements and
save to fif format
using acquisition class from brainaccess.utils

Change Bluetooth device name
"""

import matplotlib.pyplot as plt
import matplotlib
import time

from brainaccess.utils import acquisition
from brainaccess.core.eeg_manager import EEGManager

matplotlib.use("TKAgg", force=True)

eeg = acquisition.EEG()

# define electrode locations depending on your device
# halo: dict = {
#     0: "Fp1",
#     1: "Fp2",
#     2: "O1",
#     3: "O2",
# }

halo: dict = {
 0: "F4",
 1: "F3",
 2: "C4",
 3: "C3",
 4: "P4",
 5: "P3",
 6: "O2",
 7: "O1",
}

# define device name
device_name = "BA MINI 045"

# start EEG acquisition setup
with EEGManager() as mgr:
    print("with EEGManager()")
    eeg.setup(mgr, device_name=device_name, cap=halo, sfreq=250)

    # Start acquiring data
    eeg.start_acquisition()
    print("Acquisition started")
    time.sleep(3)

    start_time = time.time()
    annotation = 1
    while time.time() - start_time < 5:
        time.sleep(1)
        # send annotation to the device
        print(f"Sending annotation {annotation} to the device")
        eeg.annotate(str(annotation))
        annotation += 1

    print("Preparing to plot data")
    time.sleep(2)

    # get all eeg data and stop acquisition
    eeg.get_mne()
    eeg.stop_acquisition()
    mgr.disconnect()

# Access MNE Raw object
mne_raw = eeg.data.mne_raw
print(f"MNE Raw object: {mne_raw}")

# Access data as NumPy arrays
data, times = mne_raw.get_data(return_times=True)
print(f"Data shape: {data.shape}")

# save EEG data to MNE fif format
eeg.data.save(f'./data/{time.strftime("%Y%m%d_%H%M")}-raw.fif')
# Close brainaccess library
eeg.close()
# conversion to microvolts
mne_raw.apply_function(lambda x: x*10**-6)
# Show recorded data
mne_raw.filter(1, 40).plot(scalings="auto", verbose=False)
plt.show()