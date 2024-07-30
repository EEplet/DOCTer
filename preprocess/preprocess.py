import mne
import numpy as np
import pandas as pd
import tqdm


new_ch = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7',
          'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6',
          'CP5', 'CP6', 'FT9', 'FT10', 'TP9', 'TP10', 'F1', 'F2', 'C1', 'C2', 'P1', 'P2', 'AF3',
          'AF4', 'FC3', 'FC4', 'CP3', 'CP4', 'PO3', 'PO4', 'F5', 'F6', 'C5', 'C6', 'P5', 'P6',
          'AF7', 'AF8', 'FT7', 'FT8', 'TP7', 'TP8', 'PO7', 'PO8', 'CPz', 'POz', 'Oz', 'AFz']

old_ch = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7',
          'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6',
          'CP5','CP6', 'FT9', 'FT10', 'TP9', 'TP10', 'F1', 'F2', 'C1', 'C2', 'P1', 'P2', 'AF3',
          'AF4', 'FC3', 'FC4', 'CP3', 'CP4', 'PO3', 'PO4', 'F5', 'F6', 'C5', 'C6', 'P5', 'P6',
          'AF7', 'AF8', 'FT7', 'FT8', 'TP7', 'TP8', 'PO7', 'PO8', 'CPZ', 'POZ', 'OZ', 'FPZ']

old_ch2 = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7',
                'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6',
                'CP5', 'CP6', 'FT9', 'FT10', 'TP9', 'TP10', 'F1', 'F2', 'C1', 'C2', 'P1', 'P2', 'AF3',
                'AF4', 'FC3', 'FC4', 'CP3', 'CP4', 'PO3', 'PO4', 'F5', 'F6', 'C5', 'C6', 'P5', 'P6',
                'AF7', 'AF8', 'FT7', 'FT8', 'TP7', 'TP8', 'PO7', 'PO8', 'CPz', 'POz', 'Oz', 'Fpz']

BASEPATH = '../data_generated/allorg_2s_no_process/'

DF = pd.read_csv("./data_info.csv")

grouped_df = DF.groupby("raw_file_path")

for _i, (raw_file_path, group_df) in tqdm.tqdm(enumerate(grouped_df)):
    subject_id = group_df.loc[group_df.index.tolist()[0]]['subject_id']
    subordinate_folder = group_df.loc[group_df.index.tolist()[0]]['subordinate_folder']
    text_label = group_df.loc[group_df.index.tolist()[0]]['text_label']
    raw = mne.io.read_raw_brainvision(raw_file_path, verbose='ERROR', preload=True)

    if subordinate_folder == 'raw_data4_new':
        raw.pick_channels(new_ch)
        raw.reorder_channels(new_ch)
    elif subordinate_folder == 'raw_data4' and text_label == 'MCS':
        raw.pick_channels(old_ch2)
        raw.reorder_channels(old_ch2)
    else:
        raw.pick_channels(old_ch)
        raw.reorder_channels(old_ch)

    raw = raw.notch_filter(freqs=np.arange(50, 251, 50), verbose='ERROR')
    raw = raw.filter(l_freq=0.5, h_freq=40, verbose='ERROR')
    raw = raw.resample(sfreq=125)
    raw = raw.set_eeg_reference()

    # loop over windows
    for _j, window_idx in enumerate(group_df.index.tolist()):
        xth = group_df.loc[window_idx]['xth_2s_win']
        start_sample = xth * 2000
        stop_sample = (xth + 1) * 2000
        window_data = raw.get_data(start=start_sample, stop=stop_sample)
        file_path = BASEPATH + str(subject_id) + '_' + str(start_sample)
        print("generate: ", file_path)
        np.save(file_path, window_data)












