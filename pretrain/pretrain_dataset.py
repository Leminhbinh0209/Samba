#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 16:59:33 2021

@author: chingis
"""
import torch
from torch.utils.data import Dataset
import pandas as pd
import random
import numpy as np

class TextDataset(Dataset):
    def __init__(self, csv_file, is_train=False, max_length=500, pretrain=True):
        if '+' in csv_file:
            files = csv_file.split('+')
            df1 = pd.read_csv(files[0])
            df2 = pd.read_csv(files[1])
            df = pd.concat([df1, df2], axis=0).reset_index()
        else:
            df = pd.read_csv(csv_file)
        
        
        ids = df.ID.unique()#.values

        print('datdaset len:', len(ids))
        if pretrain:
            df =  df[df.ID.map(df.ID.value_counts()) > 1].sort_values(by=['ID'])
            print(df.ID.value_counts())
            print('datdaset len:', len(df.ID.unique()))
        self.df = df
        self.df.Transcript = self.df.Transcript.astype(str)
        self.text = self.df.Transcript.values
        self.label = self.df.Label.values
        self.idxs = np.array(list(range(len(self.df))))
        self.max_length = max_length
        self.is_train = is_train
        self.pretrain = pretrain
        if pretrain:
            self.text = self.text[::2]
            self.label = self.label[::2]
            self.idxs = self.idxs[::2]
        all_sents = zip(self.text, self.label, self.idxs)
        if self.is_train:
            all_sents = sorted(all_sents, key = lambda x : len(x[0].split()))
        self.text, self.label, self.idxs  = zip(*all_sents)
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        input_text = self.process_text(str(self.text[idx]))
        input_text  = self.curb_to_length(input_text)
        og_idx = self.idxs[idx]
        if self.pretrain:
            
            ID = self.df['ID'].iloc[og_idx]
            if (og_idx - 1 >= 0 and og_idx + 1 < len(self.df)) and (self.df['ID'].iloc[og_idx - 1] == ID and self.df['ID'].iloc[og_idx + 1] == ID):
                off = random.choice([-1, 1])
            elif (og_idx + 1 < len(self.df) and self.df['ID'].iloc[og_idx + 1] != ID) or (og_idx - 1 >= 0 and self.df['ID'].iloc[og_idx - 1] == ID):
                off = -1
            elif (og_idx - 1 >= 0 and self.df['ID'].iloc[og_idx - 1] != ID) or (og_idx + 1 < len(self.df) and self.df['ID'].iloc[og_idx + 1] == ID):
                off = 1
            sub_text = self.process_text(str(self.df.Transcript.iloc[og_idx + off]))
            sub_text  = self.curb_to_length(sub_text)
            ID2 = self.df['ID'].iloc[og_idx+ off]
            assert ID2 == ID
    
            return {'x': input_text, 'y':  self.label[idx], 'x2': sub_text, 'idx': og_idx}
        else:
            return {'x': input_text, 'y':  self.label[idx], 'idx': og_idx}
    
    def process_text(self, string):
        '''TO-DO: add pre-processing string'''
        return  string
    
    def curb_to_length(self, string):
        return ' '.join(string.strip().split()[:self.max_length])

if __name__ == '__main__':
    train = TextDataset('TRAIN.csv', True, 144)