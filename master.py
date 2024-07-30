import torch
import os
import sys
import time
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.utils.data
from operate import train_model, test_model, weight_init
from dataset.dataset import EEGDataset
import re
import random
import numpy as np
from models.DOCTer import DOCTer
import argparse
import csv
from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts
import pandas as pd
import warnings
from glob import glob
from sklearn.model_selection import StratifiedGroupKFold
import datetime
from tqdm import tqdm
from einops import rearrange, repeat, reduce
warnings.filterwarnings("ignore")

os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 
CHECKARGS = True
rej = {
    "Fp": ['Fp1', 'Fp2', "Fpz"], # Frontal Pole
    "AF": ['AF3', 'AF4', 'AF7', 'AF8'], #  Anterior Frontal
    "F" : ['F3', 'F4','F7', 'F8','F1', 'F2','F5', 'F6', 'Fz'], # Frontal
    "FT": ['FT9', 'FT10', 'FT7', 'FT8'],  # Fronto-Temporal
    "FC": ['FC1', 'FC2', 'FC5', 'FC6','FC3', 'FC4'], # Fronto-Central
    "T": ['T7', 'T8'], # Temporal
    "C": ['C3', 'C4', 'C1', 'C2','C5', 'C6', 'Cz'], # Central
    "TP": ['TP9', 'TP10','TP7', 'TP8'], # Temporal Parietal
    "CP": ['CP1', 'CP2',  'CP5', 'CP6', 'CP3', 'CP4', 'CPz'], #  Centro-Parietal
    "P": ['P3', 'P4', 'P7', 'P8', 'P1', 'P2', 'P5', 'P6','Pz'], # Parietal
    "PO":['PO3', 'PO4', 'PO7', 'PO8', 'POz'], # Parieto-Occipital
    "O": ['O1', 'O2', "Oz"], # Occipital
    "z": ['Fz', 'Cz', 'Pz', 'Fpz', 'CPz', 'POz', 'Oz']
}

def get_args():
    parser = argparse.ArgumentParser()  
    parser.add_argument('--group_index', type=int,default=1, help='group of experiment') 
    parser.add_argument('--seed', type=int, default=1999, help='seed number')  
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--testfreq', type=int, default=50, help="testfreq")
    parser.add_argument('--dropout', type=float, default=0.4, help='dropout')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight_decay')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--cuda', type=str, default="cuda:2", help='CUDA index.')
    parser.add_argument('--scheduler', type=bool, default=False, help='scheduler.')
    parser.add_argument('--timelen', type=int, default=-1, help='How many data need')  
    parser.add_argument('--fold', type=int, default=10, help='number of fold')  
    parser.add_argument('--model', type=str, default="DOCTer_7", help='model name.')  
    parser.add_argument('--datapath', type=str, default="/data0/yoro/data/allorg_2s/", help='datapath')
    parser.add_argument('--csvfile', type=str, default="./files_csv/exp.csv", help="save as csv reporting result")
    parser.add_argument('--clip', type=int, default=100, help="max_norm")
    parser.add_argument('--chs', type=str, default="all", help='filter channels')
    parser.add_argument('--ll', type=int, default=0, help='filter channels')
    parser.add_argument('--rr', type=int, default=10, help='filter channels')
    parser.add_argument('--normalize', type=str, default="y", help='filter channels')
    return parser.parse_args()

def set_seed(seed):
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed)  
    random.seed(seed)  
    np.random.seed(seed)  
    torch.backends.cudnn.benchmark = False  
    torch.backends.cudnn.deterministic = True  

def checkargs(args):
    did = "python " + os.path.basename(sys.argv[0]) + " "
    for k, v in args.__dict__.items():
        did += "--" + str(k) + " "
        did += str(v) + " "
    print(did)


args = get_args()
DEVICE = torch.device(args.cuda if torch.cuda.is_available() else 'cpu')
NORMALIZE = args.normalize
EPOCH = args.epochs 
BATCH_SIZE = args.batch_size
BASE_LR = args.lr 
WD = args.weight_decay  
DROP = args.dropout 
modelname = args.model
timelen = args.timelen
datapath = args.datapath
fold = args.fold
resfile = args.csvfile 
clip = args.clip
testfreq = args.testfreq
chs = args.chs


def setmodel(input_sample, fold_index, mean, std):
    res = {}
    if "DOCTer" in modelname:
        if modelname[-2] != "_":
            model = eval(modelname + "(drop_rate = DROP, input_sample = input_sample)")
        else:
            model = eval(modelname[:-2] + "(drop_rate = DROP, input_sample = input_sample, depth = {})".format(int(modelname[-1])))
            print("modelname:", modelname[:-2] + "(drop_rate = DROP, input_sample = input_sample, depth = {})".format(int(modelname[-1])))
        model.apply(weight_init)
    elif "unified_data_multifeature_embedding_microstate" in modelname:
        if modelname[-2] != "_":
            model = eval(modelname + "(drop_rate = DROP, input_sample = input_sample)")
        else:
            model = eval(modelname[:-2] + "(drop_rate = DROP, input_sample = input_sample, depth = {})".format(int(modelname[-1])))
            print("modelname:", modelname[:-2] + "(drop_rate = DROP, input_sample = input_sample, depth = {})".format(int(modelname[-1])))
        model.apply(weight_init)
    elif modelname == "CNN1D":
        model = CNN1D(input_sample = input_sample)
    elif modelname == "cnnlstmConformer":
        model = cnnlstmConformer(input_sample = input_sample)
    elif modelname == "EEGConformer":
        model = EEGConformer(input_sample = input_sample)
    elif modelname == "Conformer_downave":
        model = Conformer_downave(input_sample = input_sample)
    elif modelname == "Conformer_downsample":
        model = Conformer_downsample(input_sample = input_sample)
    elif modelname == "Conformer_avepool":
        model = Conformer_avepool(input_sample = input_sample)
    elif  "Conformer" in modelname:
        if "_" in modelname and modelname[-1].isdigit:
            print(modelname[:-2], "depth:", modelname[-1])
            model = Conformer(input_sample = input_sample, depth = int(modelname[-1]))
        else:
            model = Conformer(input_sample = input_sample)
    elif modelname == 'Mix':
        model = Mix(device = DEVICE, drop = DROP, input_image=input_sample)
        model.apply(weight_init)
        print("Mix weights init end~")
    elif modelname == "EEGNet":
        model = EEGNet(input_sample = input_sample)
    elif modelname == 'DCN':
        model = deepConvNet(input_sample = input_sample)
    elif modelname == "SCN":
        model = shallowConvNet(input_sample = input_sample)
    elif modelname == "TSception":
        model = TSception(input_sample = input_sample)
    else:
        assert False,"have not this model"

    model = model.to(DEVICE, non_blocking=True) 
    if "Conformer" in modelname:
        optimizer = optim.Adam(model.parameters(), lr = BASE_LR, weight_decay=WD)
    else:
        optimizer = optim.Adam(model.parameters(), lr = BASE_LR, weight_decay=WD) 
    res["model"] = model
    res["optimizer"] = optimizer
    if args.scheduler == True:
        scheduler = CosineAnnealingLR(optimizer,T_max = 64)
        res["scheduler"] = scheduler
    modelfile = "./save_models/model_"+model.__class__.__name__+"_groupid_"+str(args.group_index) +"/"
    modelpath = modelfile + "model_fork_" + str(fold_index) + '.pth'
    print(modelpath)
    if not os.path.exists(modelfile):
        os.mkdir(modelfile)
        print("create models dir!:",modelfile)
    res["modelpath"] = modelpath
    return res


def trainfunc(train_np, test_np, datapath, fold_index, mean_v, std_v):
    train_dataset =  EEGDataset(train_np, datapath, modelname, fold_index, chs, NORMALIZE, mean_v = mean_v, std_v = std_v)
    test_dataset = EEGDataset(test_np, datapath, modelname, fold_index, chs, NORMALIZE, mean_v = mean_v, std_v = std_v)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4 , pin_memory=True)
    print("loader:", len(train_loader), len(test_loader), "dataset:", len(train_dataset), len(test_dataset))
    input_sample = None
    if "DOCTer" not in modelname:
        input_sample = tuple([1] + list(test_dataset[0][0].shape))
        print("shape of data:", train_dataset[0][0].shape, "sample input:", input_sample)
    else:
        input_sample = tuple([1] + list(test_dataset[0][0][0].shape))
        print("shape of data:", train_dataset[0][0][0].shape, "sample input:", input_sample)
    config = setmodel(input_sample, fold_index, train_dataset.mean, train_dataset.std)
    model, optimizer, modelpath = config["model"], config["optimizer"], config["modelpath"]
    if args.scheduler == True:
        print("Using scheduler!")
        scheduler = config["scheduler"]
    Train_Loss_list = []
    test_Loss_list = []
    test_Accuracy_list = []

    result = {
        "test_loss": None,
        "mst_f1": 0.,
        "cmat_f1": None,
        "mst_acc": 0.,
        "cmat_acc": None,
        "mst_acc_f1_recall": None,
        "mst_precision": 0.,
        "cmat_precision": None,
        "mst_recall": 0.,
        "cmat_recall": None,
        "best res epoch": 0
    }
    display_acc = 0
    start_time = time.time()
    for epoch in range(1, EPOCH + 1):
        train_loss = train_model(model, modelname, DEVICE, train_loader, optimizer, epoch, clip, result, testfreq, test_loader, modelpath)
        Train_Loss_list.append(train_loss)
        if args.scheduler == True:
            scheduler.step()
        if display_acc < result["mst_acc"]:
            display_acc = result["mst_acc"]
          
            print("|{}-{}|result best epoch:{} loss:{} acc:{}******************************************************\n".format(modelname, args.group_index, epoch, result["test_loss"], result["mst_acc"]))
      
    print(modelpath,'Highest accuracy of validation set：{}!!!==============================================='.format(result["mst_acc"]))
    end_time = time.time()
    if not os.path.exists(resfile):
        f = open(resfile,'w',encoding='utf-8')
        f_writer = csv.writer(f)
        f_writer.writerow(("group_index","test_index", "seed", "foldn", "epochs","batch_size",
                            "dropout","weight_decay","lr","f1","f1_cmat","acc","acc_cmat","mst_acc_f1_recall",
                            "precision","precision_cmat","recall","recall_cmat","datapath","timelen", "max_norm", 
                            "testfreq","modelpath","normalize","region","best res epoch","time consumed"))
        f.close()
    
    f = open(resfile,'a',encoding='utf-8')
    f_writer = csv.writer(f)
    f_writer.writerow((args.group_index, modelname+"_"+str(fold_index), seed, fold, args.epochs, args.batch_size, args.dropout, args.weight_decay, args.lr,
        result["mst_f1"], result["cmat_f1"],
        result["mst_acc"], result["cmat_acc"], result["mst_acc_f1_recall"],
        result["mst_precision"], result["cmat_precision"],
        result["mst_recall"], result["cmat_recall"],
        datapath, timelen, clip, testfreq, modelpath, NORMALIZE, chs, result["best res epoch"], int(end_time-start_time)))
    f.close()
    print("this fold run done", "!"*40)
    return result["mst_acc"]

def get_meanstd_merge(file_paths, fold_index, path):
    t_begin = time.time()
    meanfilename = path + "mean_" + str(fold_index) + ".npy"
    stdfilename = path + "std_" + str(fold_index) + ".npy"
    print("fold_index{}: number of files|{}|:".format(fold_index, path), len(file_paths))
    shape = np.load(path + file_paths[0], allow_pickle=True).item()["eeg"].shape# .shape
    print("number of channels:", shape)
    sum = np.zeros([shape[0],])
    timelen = shape[1]
    for fp in tqdm(file_paths):
        if "mean" in fp or "std" in fp:
                continue
        da = np.load(path + fp, allow_pickle=True).item()["eeg"]
        sum += np.sum(da, axis = 1)
    mean = sum / (len(file_paths)*timelen)
    t_mean = repeat(mean, "n -> n t", t = timelen)
    sumstd = np.zeros(list(shape))
    for fp in tqdm(file_paths):
        if "mean" in fp or "std" in fp:
            continue
        da = np.load(path + fp, allow_pickle=True).item()["eeg"]
      
        sumstd += (da - t_mean)**2 
    std = (np.sum(sumstd, axis = 1)/(len(file_paths)*timelen))**0.5
    print("********************************************************mean:\n", mean)
    print("********************************************************std:\n", std)
    print("---------------------------------------------------------------------time:", time.time()-t_begin)
    return mean, std

def trainfold(path, foldn = 10):
    f_paths = glob(path + '**.npy', recursive=True)
    f_paths.sort()  # !!!
    mean_flag = True
    if path + "mean.npy" in f_paths:
        f_paths.remove(path + "mean.npy")
        f_paths.remove(path + "std.npy")

    for i in range(10):
        if path + "mean_{}.npy".format(i) in f_paths:
            f_paths.remove(path + "mean_"+str(i)+".npy")
            f_paths.remove(path + "std_"+str(i)+".npy")
        else:
            mean_flag = False
    print("remove mean&std_fold_test")
    X = []
    y = []
    groups = []
    for fp in f_paths:
        fname = fp.split("/")[-1]

        if 'uws' in fp:
            res = re.findall("uws[1-9]\\d*", fname)
        elif 'mcs' in fp:
            res = re.findall("mcs[1-9]\\d*", fname)
        else:
            assert 0, "error! cann't found mcs and uws"
       
        X.append(fname)
        y.append(int(fname[0]))
        groups.extend(res) 
    print("all data number debug:", len(X), len(y), len(groups), len(set(groups)))

    sgkf = StratifiedGroupKFold(n_splits=foldn, shuffle=True, random_state=99) 
    ave_acc = 0.
    
    for fold_index, (train, test) in zip(range(foldn), sgkf.split(X, y, groups=groups)):
        print("fold_index:%d trainlen:%d traintest:%d=>%s" % (fold_index, len(train), len(test), test))
        train_np, test_np = [], []
        train_groups, test_groups = [], []
        for idx in train:
            if timelen == -1:          
                train_np.append(X[idx])
                train_groups.append(groups[idx])
            elif int(re.findall("[0-9]\\d*", X[idx])[-1]) < timelen:
              
                train_np.append(X[idx])
        for idx in test:                
            test_np.append(X[idx])
            test_groups.append(groups[idx])
        print("data number check: test-", len(test_np), "train-", len(train_np), set(train_np).intersection(set(test_np))) 
        
        mean_v, std_v = None, None
        if "merge" in path and not mean_flag:
            mean_v, std_v = get_meanstd_merge(train_np, fold_index, path)

        ave_acc += trainfunc(train_np, test_np, path, fold_index, mean_v, std_v)
        set_seed(seed) 

    ave_acc /= foldn
    print("Average acc:", ave_acc)

if __name__ == '__main__':
    if CHECKARGS == True:
        checkargs(args)
    seed = args.seed 
    set_seed(seed) 
    print("result csv file:", resfile)
    trainfold(datapath, foldn = fold)