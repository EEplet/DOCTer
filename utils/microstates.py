import numpy as np
import pandas as pd
import mne
from pycrostates.cluster import ModKMeans
import os
import copy
import tqdm
import json
import pickle

df = pd.read_csv("../data_info.csv")  # read data info
df = df[(df['text_label'] == 'UWS') | (df['text_label'] == 'MCS')]
df = df[df['reject'] == 0]
df = df.reset_index(drop=True)
for i, (subject_id, group_df) in tqdm.tqdm(enumerate(df.groupby("subject_id"))):
    slice_list = []
    for index, row in group_df.iterrows():
        start_sample = row['start_sample_index']
        win_name = str(subject_id) + '_' + str(start_sample)
        win_data = np.load('../data/' + win_name + '.npy')
        slice_list.append(copy.deepcopy(win_data))
    sbj_eeg = np.concatenate(slice_list, axis=1)
    sbj_eeg = sbj_eeg[:62, :]
    info = mne.create_info(
        ch_names=['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7',
                  'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5',
                  'CP6', 'FT9', 'FT10', 'TP9', 'TP10', 'F1', 'F2', 'C1', 'C2', 'P1', 'P2', 'AF3', 'AF4', 'FC3',
                  'FC4', 'CP3', 'CP4', 'PO3', 'PO4', 'F5', 'F6', 'C5', 'C6', 'P5', 'P6', 'AF7', 'AF8', 'FT7', 'FT8',
                  'TP7', 'TP8', 'PO7', 'PO8', 'CPz', 'POz', 'Oz'],
        ch_types='eeg', sfreq=1000/8)
    raw = mne.io.RawArray(sbj_eeg, info)
    raw.set_montage('standard_1005')
    ModK = ModKMeans(n_clusters=4, random_state=21)
    ModK.fit(raw, n_jobs=8)
    with open("./template.pickle", "rb") as f:
        micro_temp = pickle.load(f)
    ModK.reorder_clusters(template=micro_temp)
    ModK.rename_clusters(new_names=["A", "B", "C", "D"])
    segmentation = ModK.predict(
        raw,
        factor=10,  # Factor used for label smoothing.
        half_window_size=5,
        min_segment_length=5,
    )
    label = np.array(segmentation.labels)
    save_path = '/data2/' + 'microstates/' + f'{subject_id}'
    np.save(save_path + '.npy', label)
    parameters = segmentation.compute_parameters()
    with open(save_path + ".json", 'w') as f:
        json.dump(parameters, f)








