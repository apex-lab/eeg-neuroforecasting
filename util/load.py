import numpy as np 
import pandas as pd
import mne 
import re
import os

def load_avr(filepath, return_channels = False):
    '''
    Reads an .avr file (as exported by BESA) into a pandas dataframe

    Currently assumes you're using an EGI net. Sorry.

    TO-DO: apparently sometimes BESA outputs .avr files without a line for
    electrode labels (I guess if you didn't feed it a layout file), so it
    could be useful to implement a conditional for that case. Instead, 
    function will throw an error if data dimensions don't match the 
    metadata, so it will just reject .avr files without electrode names.

    Sometimes, if current montage is exported, BESA will output virtual
    EOG channels and artifact channels. This function removes those,
    assuming they come after the real electrodes. If they don't for
    some reason, mistakes may be made.
    '''

    # read metadata in first line seperately and put in a dictionary
    f = open(filepath)
    md = f.readline()
    md = re.findall(r'\S+', md) # split string by white space
    metadata = {}
    for i in range(len(md)):
        if i % 2 == 0: # a.k.a. skip every other index
            field_name = md[i][:-1] # last character is always '=' so we remove
            val = float(md[i + 1])
            metadata[field_name] = val

    # read electrode names manually since not aligned to columns
    en = f.readline()
    electrode_names = re.findall(r'(E\d+|Cz)', en)

    # and then read the data into a dataframe
    data = pd.read_table(f, delim_whitespace = True, header = None) 
    data = data.transpose() # so electrodes can be columns now
    data = data * 1e-6 # scale to match MNE's default units

    # if more electrode names than columns, then re failed because each label
    # was followed by '-<ref>', so get rid of the extra entries
    electrode_names = [chan for chan in electrode_names if chan != "E"]
    if len(electrode_names) > data.shape[1]:
        electrode_names = electrode_names[::2]
    data = data.iloc[:, :len(electrode_names)]

    # catch error is channels names are missing from second line
    if data.shape[0] != metadata["Npts"]:
        raise Exception("Dimensions of loaded data don't match metadata" + 
            " for %s. Does the second line contain channel names?"%filepath)
    if return_channels:
        return data, metadata, electrode_names
    else:
        return data, metadata


def read_evoked_avr(filepath, montage, condition_name = None, highpass = None,
    lowpass = None, custom_ref = None, bad_channels = None):
    '''
    Loads an .avr file into MNE Python as an mne.EvokedArray object
    '''
    df, metadata, channels = load_avr(filepath, return_channels = True) 
    fs = 1/metadata["DI"] * 1000
    start_t = metadata["TSB"]/1000
    info = mne.create_info(channels, fs, ch_types = 'eeg', verbose = None)
    info.set_montage(montage)
    if highpass is not None:
        info["highpass"] = highpass # float or int
    if lowpass is not None:
        info["lowpass"] = lowpass # float or int
    if custom_ref is not None:
        info["custom_ref_applied"] = custom_ref # a boolean is fine
    if bad_channels is not None:
        info["bads"] = bad_channels # list of strings
    if condition_name is not None:
        info["description"] = condition_name # string
        evoked_array = mne.EvokedArray(np.transpose(df), info, start_t, comment = condition_name)
    else:
        evoked_array = mne.EvokedArray(np.transpose(df), info, start_t)
    return evoked_array

def read_epochs_avr(directory, montage, condition, subject = None,
    highpass = None, lowpass = None, custom_ref = None, bad_channels = None):
    '''
    Loads a directory of .avr files into an mne.EpochsArray object

    condition requires an (int, str) = (condition number, condition name) tuple
    if only one condition, or a list of (int, str) tuples. Files will be 
    processed in alphanumeric order, so list of tuples should be in the 
    order of the alphanumerically sorted files.
    '''
    # catch a predictable mistake
    if isinstance(condition, list) and isinstance(condition[0], int):
        raise ValueError("For single condition, must input an (int, str) tuple" + 
            " NOT an [int, str] list!")
    # collect data from across files
    metadata = None
    channels = None
    data = []
    files = os.listdir(directory)
    for f in sorted(files):
        path = (os.path.join(directory, f))
        df, metadata, channels = load_avr(path, return_channels = True)
        data.append(np.transpose(df))
    data = np.stack(data, axis = 0)
    # make info structure
    fs = 1/metadata["DI"] * 1000
    start_t = metadata["TSB"]/1000
    info = mne.create_info(channels, fs, ch_types = 'eeg', verbose = None)
    info.set_montage(montage)
    if highpass is not None:
        info["highpass"] = highpass # float or int
    if lowpass is not None:
        info["lowpass"] = lowpass # float or int
    if custom_ref is not None:
        info["custom_ref_applied"] = custom_ref # str or bool is fine
    if bad_channels is not None:
        info["bads"] = bad_channels # list of strings
    info["description"] = "sub-" + str(subject)
    # make condition list
    events = []
    event_id = {}
    for trial in range(data.shape[0]):
        if isinstance(condition, list):
            cond = condition[trial]
        else:
            cond = condition
        events.append([trial, 0, cond[0]])
        event_id[cond[1]] = cond[0]
    events = np.array(events)
    epochs_array = mne.EpochsArray(data, info, events, start_t, event_id)
    return epochs_array

def load(avr_directory, subject, conditions):
    '''
    A loading utility idiosyncratic to this project. Navigates 'exports'
    directory structure to load files for a given subject and 
    conditions (e.g. ["funded", "unfunded"])

    Then, re-references to average before returning

    Expects a directory structure like avr_directory/subject/condition/*.avr
    '''
    data = []
    cond_count = 1
    for cond in conditions:
        path = os.path.join(avr_directory, subject, cond)
        try:
            epochs_cond = read_epochs_avr(path, "GSN-HydroCel-129", 
                (cond_count, cond), highpass = .3, lowpass = 100)
            data.append(epochs_cond)
        except:
            print("No %s files for sub-%s"%(cond, subject))
        cond_count += 1
    epochs = mne.concatenate_epochs(data, add_offset = True)
    epochs.set_eeg_reference('average', projection = False)
    return epochs
























