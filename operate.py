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
from tqdm import tqdm 


def weight_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
        init.kaiming_normal_(m.weight.data)
        init.constant_(m.bias.data,0.1)
   
    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0,0.01)
        m.bias.data.zero_()


def train_model(model, modelname, device, train_loader, optimizer, epoch, clip, result, testfreq, test_loader, modelpath, display = True, testdisplay = False):
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    for batch_index, (data, label) in enumerate(tqdm(train_loader, desc = "Training")): 
        label = label.to(device, non_blocking=True)
        if "DOCTer" in modelname:
            x, micro_sequence, cause_number, bc_number = data
            x = x.to(device, non_blocking=True)
            micro_sequence = micro_sequence.to(device, non_blocking=True)
            cause_number = cause_number.to(device, non_blocking=True)
            bc_number = bc_number.to(device, non_blocking=True)
            data = (x, micro_sequence, cause_number, bc_number)
        else:
            data = data.to(device, non_blocking=True) 
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)
        loss.backward()


        if clip < 50:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = clip) 
        
        optimizer.step()
        
        if batch_index == 0:
            continue
        if display and batch_index % 20 == 0:  
            tqdm.write('==========================Train Epoch:{} |batch_index:{} |loss:{:.6f} end!=========================='.format(epoch, batch_index, loss.item())) 
        if (batch_index % testfreq == 0 or batch_index == len(train_loader) - 1)  and not (test_loader is None):
            test_loss, test_f1, test_acc, test_precision, test_recall, cmat = test_model(model, modelname, device, test_loader, display = testdisplay) 
            result["test_loss"] = test_loss
            if test_acc > result["mst_acc"]:                                                                    
                result["mst_acc"], result["cmat_acc"] = test_acc, cmat
                result["mst_acc_f1_recall"] = {"f1": test_f1, "recall": test_recall}
                result["best res epoch"] = epoch
                torch.save(model, modelpath)
            if test_f1 > result["mst_f1"]:
                result["mst_f1"], result["cmat_f1"] = test_f1, cmat
            if test_precision > result["mst_precision"]:
                result["mst_precision"], result["cmat_precision"] = test_precision, cmat
            if test_recall > result["mst_recall"]:
                result["mst_recall"], result["cmat_recall"] = test_recall, cmat
            model.train()   

    train_loss = 0
    return train_loss  

def test_model(model, modelname, device, loader, display = True):
    model.eval()
    loss_res = 0.0
    criterion = torch.nn.CrossEntropyLoss(reduction='sum') 
    y_true, y_pred = torch.tensor([]).to(device), torch.tensor([]).to(device)

    with torch.no_grad():   
        for data, label in loader:
            label = label.to(device, non_blocking=True)
            if "unified_data_multifeature_embedding_microstate" in modelname:
                x, micro_sequence, cause_number, bc_number = data
                x = x.to(device, non_blocking=True)
                micro_sequence = micro_sequence.to(device, non_blocking=True)
                cause_number = cause_number.to(device, non_blocking=True)
                bc_number = bc_number.to(device, non_blocking=True)
                data = (x, micro_sequence, cause_number, bc_number)
            else:
                data = data.to(device, non_blocking=True)
            output = model(data)
            
            loss = criterion(output, label)
            loss_res += loss.item()

            pred = output.argmax(dim=1)  
            y_true = torch.cat((y_true, label), dim=0)
            y_pred = torch.cat((y_pred, pred), dim=0)
    y_true, y_pred = y_true.to("cpu", non_blocking=True).numpy(), y_pred.to("cpu", non_blocking=True).numpy()
    f1, acc, precision, recall, cmat = f1_score(y_true,y_pred),accuracy_score(y_true,y_pred),precision_score(y_true,y_pred),recall_score(y_true,y_pred), confusion_matrix(y_true, y_pred)
    
    loss_res /= len(loader.dataset)  
    if display == True:
        tqdm.write("Test situation:  loss:{:.6f}| f1_score:{:.6f}| accuracy_score:{:.6f}| precision_score:{:.6f}| recall_score:{:.6f} ".format(loss_res,f1,acc,precision,recall))   
    return loss_res, f1, acc, precision, recall, cmat