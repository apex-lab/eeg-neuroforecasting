import numpy as np
import mne
import re
import os


def load_conds(conds, epochs_dir, include_info = False):
    # get subject files
    fnames = [f for f in os.listdir(epochs_dir) if conds in f]
    subj_nums = [re.search("\d+", f).group(0) for f in fnames]

    # load in data
    eeg = []
    subj_idx = [] # keeps track of which trial belongs to which subj
    trial_cts = []
    for i in range(len(subj_nums)):
        epochs = mne.read_epochs(os.path.join(epochs_dir, fnames[i]), 
            verbose = False)
        epochs.crop(-.1, .8)
        n_trials = epochs.events.shape[0]
        subj_idx += n_trials*[subj_nums[i]]
        trial_cts.append(n_trials)
        eeg.append(epochs)
    eeg = mne.concatenate_epochs(eeg)

    # and format for classification
    X = eeg.get_data() 
    y = eeg.events[:, 2]
    y = np.where(y == 2, 0, y) # so funded/yes = 1 and unfunded/no = 0

    if include_info:
        return X, y, subj_idx, epochs.times, epochs.info
    else:
        return X, y, subj_idx, epochs.times