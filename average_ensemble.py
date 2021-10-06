import os
import statistics
import json
import pandas as pd
import sys
import collections
import pickle
from time import time
from datetime import timedelta
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import  DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from easydict import EasyDict as edict
import yaml

def main(config):
    uscs_dir = f"{config.data_folder}/{config.dataset}_data/" 
    uscs_big_subtitle = pd.read_csv(uscs_dir + f"transcripts/{config.dataset.lower()}_big_subtitle.csv")
    target = uscs_big_subtitle.label.values
        
    # Use K fold split by channel ID
    with open(uscs_dir + "/meta_data/k_fold_channel.json", 'rb') as fp:
        print("Load K FOLD by CHANNEL")
        k_fold_channel = pickle.load(fp)

    all_methods = [ 'RF', 'SVM', 'LR',  'aaai' ,'TREE', 'NB', 'KN', 'CNN',] 
    dataset_pred_binary = np.zeros_like(target)
    feature_importances = None
    test_true = np.array([]) 
    test_pred = np.array([])
    for fold in range(5):
        print(f"FOLD: {fold+1}")
        train_val_set_indices, test_set_indices = k_fold_channel[fold]['train'], k_fold_channel[fold]['test']
        Y_train = np.take(target, indices=train_val_set_indices, axis=0)
        Y_test = np.take(target, indices=test_set_indices, axis=0)

        train_data = np.zeros(shape=(len(Y_train), len(all_methods)))
        test_data = np.zeros(shape=(len(Y_test), len(all_methods)))
        for m_idx, method in enumerate(all_methods):
            in_folder = f"{uscs_dir}/word2vec/{method}/fold-{fold+1}/"
            if method == 'aaai':
                in_folder = f"{uscs_dir}/meta_data/{method}/fold-{fold+1}/"
                train_m = np.load(f"{in_folder}/train.npy")
                test_m = np.load(f"{in_folder}/test.npy")[:,1]
            else:
                train_m = np.load(f"{in_folder}/train.npy")
                test_m = np.load(f"{in_folder}/test.npy")
            train_data[:, m_idx] = train_m
            test_data[:, m_idx] = test_m
            print(method, accuracy_score(1-Y_test, (1-(test_m > 0.5).astype(np.int))))
        print("Averaging accuracy: ", accuracy_score(1-Y_test, (1-(np.mean(test_data, axis=1) > 0.5).astype(np.int))))
        dataset_pred_binary[test_set_indices] = np.mean(test_data, axis=1).flatten() #test_pred
        print(len( np.mean(test_data, axis=1)))
        print(np.mean(test_data, axis=1).shape)
        test_true = np.hstack((test_true, Y_test)) 
        test_pred = np.hstack((test_pred, np.mean(test_data, axis=1)))
        assert len(Y_test) == len(test_data), "Lenght miss - align"
        # print(len(test_true), len(test_pred))
        # # Fit the model
        # model = LogisticRegression(fit_intercept=False)
        # # model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=16, random_state=41)
        # model.fit(X=train_data, y=Y_train)
 
    print("\033[92m")
    AVERAGE_USED = 'macro'
    target = test_true
    dataset_pred_binary = (test_pred > 0.5).astype(np.int)
    test_accuracy = accuracy_score(1-target, 1-dataset_pred_binary)
    test_precision = precision_score(1-target, 1-dataset_pred_binary, average=AVERAGE_USED)
    test_recall = recall_score(1-target, 1-dataset_pred_binary, average=AVERAGE_USED)
    test_f1_score = f1_score(1-target,1-dataset_pred_binary, average=AVERAGE_USED)

    print('--- TEST Accuracy: {:.3f}'.format(test_accuracy))
    print('--- TEST Precision: {:.3f}'.format(test_precision))
    print('--- TEST Recall: {:.3f}'.format(test_recall))
    print('--- TEST F1-Score: {:.3f}'.format(test_f1_score))
    print('\033[0m')

if __name__ == '__main__':
    with open('./config/config_pre.yaml', 'r') as stream:
        config = yaml.safe_load(stream)
    config = edict(config)
    
    main(config)
