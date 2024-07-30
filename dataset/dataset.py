import torch
import os 
from torch import nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import re
import numpy as np
from torch.nn import init
from sklearn.metrics import f1_score,accuracy_score,precision_score,recall_score, confusion_matrix
from einops import rearrange, repeat
from scipy import signal


def mask(rej_name):
    ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 
    'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'FC1', 'FC2', 
    'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6', 'FT9', 'FT10', 'TP9', 'TP10', 'F1', 
    'F2', 'C1', 'C2', 'P1', 'P2', 'AF3', 'AF4', 'FC3', 'FC4', 'CP3', 'CP4', 'PO3', 
    'PO4', 'F5', 'F6', 'C5', 'C6', 'P5', 'P6', 'AF7', 'AF8', 'FT7', 'FT8', 'TP7', 
    'TP8', 'PO7', 'PO8', 'Fpz', 'CPz', 'POz', 'Oz']
    rej_chs = rej[rej_name]
    result = []
    for i in range(len(ch_names)):
        if ch_names[i] not in rej_chs:
            result.append(i)
    assert len(result) + len(rej_chs) == len(ch_names), "ERROR"
    return result

region = {
    'LA': [0, 2, 10, 19, 23, 27, 31, 37, 39, 45, 51, 53],
    'RA': [1, 3, 11, 20, 24, 28, 32, 38, 40, 46, 52, 54],
    'LP': [6, 8, 14, 21, 25, 29, 35, 41, 43, 49, 55, 57],
    'RP': [7, 9, 15, 22, 26, 30, 36, 42, 44, 50, 56, 58],
    'LALP': [0, 2, 4, 6, 8, 10, 12, 14, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57],
    'RARP': [1, 3, 5, 7, 9, 11, 13, 15, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58],
    'LARA': [0, 1, 2, 3, 10, 11, 16, 19, 20, 23, 24, 27, 28, 31, 32, 37, 38, 39, 40, 45, 46, 51, 52, 53, 54],
    'LPRP': [6, 7, 8, 9, 14, 15, 18, 21, 22, 25, 26, 29, 30, 35, 36, 41, 42, 43, 44, 49, 50, 55, 56, 57, 58, 59, 60, 61],
    'Center': [4, 5, 12, 13, 16, 17, 18, 33, 34, 47, 48, 59, 60, 61],
    'LALPRA': [0, 2, 10, 19, 23, 27, 31, 37, 39, 45, 51, 53, 6, 8, 14, 21, 25, 29, 35, 41, 43, 49, 55, 57, 1, 3, 11, 20, 24, 28, 32, 38, 40, 46, 52, 54],
    'CH08': [0, 1, 4, 5, 8, 9, 12, 13],
    'CH16': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    'CH32': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 37, 38, 43, 44, 61]
}

class EEGDataset(Dataset):
    """EEGLearn Images Dataset from EEG."""
    
    def __init__(self, data_list, path, dtype, fold_index, chs = "all", normalize = "y", mean_v = None, std_v = None):                                 
        self.normalize = normalize
        self.data_list = data_list
        self.path = path
        if "merge" in path:
            self.infopath = None
        elif "data02s" in path:
            self.infopath = path[:-1] + "_info/"
            print("infopath:", self.infopath, "*=_" * 20)
        elif "_2s" in path:
            self.infopath = "/data/allorg_2s_other_info/"
        elif "_04s" in path:
            self.infopath = "/data/allorg_4s_other_info/"
        elif "_08s" in path:
            self.infopath = "/data/allorg_8s_other_info/"
        elif "_12s" in path:
            self.infopath = "/data/allorg_12s_other_info/"
        elif "_16s" in path:
            self.infopath = "/data/allorg_16s_other_info/"
        elif "_20s" in path:
            self.infopath = "/data/allorg_20s_other_info/"
        else:
            assert 0, "no info path"

        self.dtype = dtype
        if mean_v is None and std_v is None:
            self.mean = np.load(path + "mean_" + str(fold_index) + ".npy") 
            self.std = np.load(path + "std_" + str(fold_index) + ".npy")
        else:
            self.mean = mean_v
            self.std = std_v
            
        self.index = None
        self.chs = chs
        if dtype == "Mix":
            shape = np.load(self.path + data_list[0]).shape  #
            tlen, hlen, wlen = shape[0], shape[2], shape[3]
            self.mean = repeat(self.mean, "c -> t c h w", t = tlen, h = hlen, w = wlen)
            self.std = repeat(self.std, "c -> t c h w", t = tlen, h = hlen, w = wlen)
        else:
            if "merge" not in path:
                tlen = np.load(self.path + data_list[0]).shape[-1]  #
            else:
                tlen = np.load(self.path + data_list[0], allow_pickle=True).item()["eeg"].shape[-1]
            self.mean = repeat(self.mean, "n -> n t", t = tlen)[:-1, :]   # Discard the last channel
            self.std = repeat(self.std, "n -> n t", t = tlen)[:-1, :]
        if dtype == "TSception":
            l = [0, 2, 4, 6, 8, 10, 12, 14, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57]
            r = [1, 3, 5, 7, 9, 11, 13, 15, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58]
            self.index = l + r
        if chs != "all":
            if len(chs) == 2:
                # self.index = [idx for idx in range(62) if idx not in region[chs]]  # Exclude brain regions
                self.index = [idx for idx in range(62) if idx in region[chs]]
            elif len(chs) == 4:
                self.index = region[chs]
                print(chs+":", self.index)
          
        

    def __len__(self):
        return len(self.data_list)   
    
    def __getitem__(self, idx):
        label = int(self.data_list[idx][0])
    
        if "Conformer" in self.dtype or self.dtype == 'CNN1D':
            out = np.load(self.path + self.data_list[idx])[:-1, :]
            out = (out - self.mean)/self.std
            out = out[:, ::8]  
            out = rearrange(out, "c t -> 1 c t")
            out = torch.as_tensor(out, dtype = torch.float)
        elif "DOCTer" in self.dtype:
            if "merge" in self.path:
                data = np.load(self.path + self.data_list[idx], allow_pickle=True).item()
                out = data["eeg"][:-1, :]
                other_info = data["mic"]
            else:
                out = np.load(self.path + self.data_list[idx])[:-1, :]
                other_info = np.load(self.infopath + self.data_list[idx])
            micro_number, cause_number, bc_number = other_info[:-2], other_info[-2], other_info[-1]
            micro_number = torch.as_tensor(micro_number[::8], dtype = torch.float)
            if self.normalize == "y":
                out = (out - self.mean)/self.std
            out = out[:, ::8]  
            if self.chs != "all":
                out = np.take(out, self.index, axis = 0)
            out = torch.as_tensor(out, dtype = torch.float)
            out = (out, micro_number, cause_number, bc_number)

        elif self.dtype == "TSception":
            out = np.load(self.path + self.data_list[idx])[:-1, :]
            out = (out - self.mean)/self.std
            out = out[:, ::4] # sample_rate = 250
            out = np.take(out, self.index, axis = 0)
            out = rearrange(out, "c t -> 1 c t")
            out = torch.as_tensor(out, dtype = torch.float)
      
        elif self.dtype == "RCNN":
            out = np.load(self.path + self.data_list[idx])
            out = (out - self.mean)/self.std
            out = torch.as_tensor(out, dtype = torch.float)

        elif self.dtype=='EEGNet' :
            out = np.load(self.path + self.data_list[idx])[:-1, :]
            out = (out - self.mean)/self.std
            out = out[:, ::8]
            out = torch.as_tensor(out, dtype = torch.float)
        else:
            assert 0, "Don't have!"
        sample = (out, label)
        return sample
