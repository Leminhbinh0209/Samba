#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 13:31:31 2021

@author: chingis
"""
import argparse
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from model import BertClassifier
from pretrain_dataset import TextDataset
import torch.nn as nn
from tqdm import tqdm
import torch.multiprocessing
import moco.builder
torch.multiprocessing.set_sharing_strategy('file_system')
parser = argparse.ArgumentParser(description='Document embedding')

parser.add_argument('--document', metavar='DIR', default='./datasets/TRAIN.csv',
                    help='path to dataset')
parser.add_argument('--save', metavar='DIR', default='./train_moco',
                    help='save folder')
parser.add_argument('--checkpoint', metavar='DIR', default='./encoder_q.tar',
                    help='save folder')
args = parser.parse_args()
BATCH_SIZE = 1024
document = args.document
save_folder = args.save
cp = args.checkpoint

dataset = TextDataset(document, False, 144, False)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

device = 'cuda'
model = BertClassifier(device = device)
model.load_state_dict(torch.load(cp)['net'])
model.to(device)

documents = {'idx':[],
             'X':[],
             'Y':[],
             }
model.eval()
with torch.no_grad():
    for batch_num, data in enumerate(tqdm(dataloader)): 
        x_batch = data['x'].to(device, non_blocking=True)
        y_batch = data['y'].long()
        idx = data['idx']

        emb_x = model(x_batch)

        documents['idx'].append(idx.detach().cpu())
        documents['X'].append(emb_x.detach().cpu())
        documents['Y'].append(y_batch.detach().cpu())


    documents['idx'] =  torch.cat(documents['idx'], 0).numpy()
    documents['X'] =torch.cat(documents['X'], 0).numpy()
    documents['Y'] =torch.cat(documents['Y'], 0).numpy()

print(documents['X'].shape)

print('Processing data')
total = {}
for idx, x, y in tqdm(zip(documents['idx'], documents['X'], documents['Y'])):
    ID = dataset.df['ID'].iloc[idx]
    x = x.reshape(1, 768)
    if ID not in total.keys():
        total[ID] = {'x' : [x], 'y':y}
    else:
        total[ID]['x'].append(x)

print('Saving data')
max_len = 0
min_len = float('inf')
for ID in tqdm(total.keys()):
    total[ID]['x'] = np.concatenate(total[ID]['x'], 0)
    assert total[ID]['x'].shape[1] == 768
    max_len = max(max_len, len(total[ID]['x']))
    min_len = min(min_len, len(total[ID]['x']))
    data = [ID, total[ID]['x'], total[ID]['y']]
    pickle.dump(data, open(f'{save_folder}/{ID}.bin', 'wb'))
print(max_len)
print(min_len)
