import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

# Import General Libraries
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from sklearn.feature_extraction.text import CountVectorizer
# from keras.preprocessing.text import Tokenizer
from tensorflow.keras import layers

try: 
    from cuml.svm import SVC # Cuda-supported SVM
except:
    print("Can not import the cuda SVM version ")
    from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import  DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.metrics import roc_curve, auc, accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
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
from easydict import EasyDict as edict
import yaml
import string
from tensorflow.keras.utils import to_categorical
import re

os.environ ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 


def stratified_train_val_split(set_labels, val_size=0.2):
    # Declare vaiables
    indices_train, indices_val = list(), list()

    # Get the indices of the Appropriate videos
    indices_appropriate = [i for i,x in enumerate(set_labels) if x == 0]

    # Get the indices of the Disturbing videos
    indices_disturbing = [i for i, x in enumerate(set_labels) if x == 1]
    # APPROPRIATE
    total_appropriate_train = int(len(indices_appropriate) * (1 - val_size))
    indices_train = indices_appropriate[0:total_appropriate_train]
    indices_val = indices_appropriate[total_appropriate_train:len(indices_appropriate)]

    # DISTURBING
    total_disturbing_train = int(len(indices_disturbing) * (1 - val_size))
    indices_train += indices_disturbing[0:total_disturbing_train]
    indices_val += indices_disturbing[total_disturbing_train:len(indices_disturbing)]

    print('--- [TRAIN_VAL_SPLIT] TOTAL VIDEOS: %d | TOTAL TRAIN: %d, TOTAL VAL: %d' % (len(set_labels), len(indices_train), len(indices_val)))
    return indices_train, indices_val


def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label

def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "\'", "'")
    return tf.strings.regex_replace(
        stripped_html, "[%s]" % re.escape(string.punctuation), ""
    )

def CNN_model(train_tokens):
    max_features = 20000
    embedding_dim = 128
    sequence_length = 5000
    vectorize_layer = TextVectorization(
                    standardize=custom_standardization,
                    max_tokens=max_features,
                    output_mode="int",
                    output_sequence_length=sequence_length,
                    )
    vectorize_layer.adapt(train_tokens)
    # 'embedding_dim'.
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(1,), dtype=tf.string))
    model.add(vectorize_layer)
    model.add(layers.Embedding(max_features, embedding_dim))
    model.add(layers.Conv1D(28, 9, padding="same", activation="relu", strides=3))
    model.add(layers.Conv1D(28, 9, padding="same", activation="relu", strides=3))
    model.add(layers.Conv1D(28, 9, padding="same", activation="relu", strides=3))
    model.add(layers.GlobalMaxPooling1D())

    model.add(layers.Dense(28, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation="sigmoid", name="predictions"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

def main(config):
    ### Read and split dataset
    youtube_dir = f"{config.data_folder}/{config.dataset}_data/" 
    youtube_big_subtitle = pd.read_csv(youtube_dir + f"transcripts/{config.dataset.lower()}_big_subtitle.csv")
    target = youtube_big_subtitle.label.values
    
    # Word2Vec Embedding 
    with open(youtube_dir + "/transcripts/word2vec_embedding.json", 'rb') as fp:
        word2vec_dict = pickle.load(fp)
 
 
    # For final evaluation
    dataset_pred_binary = np.zeros_like(target)
    tic = time()

    out_folder = f"{youtube_dir}/word2vec/{config.method}//"
    os.makedirs(out_folder, exist_ok=True)

    f = open(f"{youtube_dir}{config.dataset.lower()}_train_videos.txt", "r")
    train_video_id = f.readlines()
    train_video_id = [i.strip() for i in train_video_id]
    f.close()

    f = open(f"{youtube_dir}{config.dataset.lower()}_test_videos.txt", "r")
    test_video_id = f.readlines()
    test_video_id = [i.strip() for i in test_video_id]
    f.close()

    index_lookup = dict(zip(youtube_big_subtitle.video_id.values, np.arange(len(youtube_big_subtitle))))
    train_val_set_indices = [index_lookup[u] for u in train_video_id if u in index_lookup]
    test_set_indices = [index_lookup[u] for u in test_video_id if u in index_lookup]

    Y_train = np.take(target, indices=train_val_set_indices, axis=0)
    Y_test = np.take(target, indices=test_set_indices, axis=0)
    X_train = word2vec_dict["train"]
    X_test = word2vec_dict["test"]
    print("=== Model training infor ===")
    print("Number training samples: ", X_train.shape[0], len(Y_train))
    print("Number test samples: ", X_test.shape[0], len(Y_test))

    if config.method  =="CNN":
        train_val_tokens = youtube_big_subtitle.loc[train_val_set_indices, 'subtitle'].values
        print(youtube_big_subtitle.head())
        print(train_val_tokens[0:])
        
        test_tokens = youtube_big_subtitle.loc[test_set_indices, 'subtitle'].values.tolist()
        # train test
        Y_train_val = np.take(target, indices=train_val_set_indices, axis=0)
        Y_test = np.take(target, indices=test_set_indices, axis=0)

        # train - val
        indices_train, indices_val = stratified_train_val_split(set_labels=Y_train, val_size=0.2)
        # Get Y_train and Y_val
        train_tokens = np.take(train_val_tokens, indices=indices_train, axis=0).tolist()
        Y_train = np.take(Y_train_val, indices=indices_train, axis=0).tolist()
        val_tokens = np.take(train_val_tokens, indices=indices_val, axis=0).tolist()
        Y_val = np.take(Y_train_val, indices=indices_val, axis=0).tolist()
        print(train_tokens[:3])
        print(Y_train[:3])
        model = CNN_model(train_tokens)
        early_stopper = EarlyStopping(mode='auto', verbose=2, monitor='val_loss', restore_best_weights=True, patience=10)
        model.fit(train_tokens, Y_train, 
                    batch_size=config.batch_size, 
                    validation_data=(val_tokens, Y_val), 
                    epochs=50,
                    callbacks=[early_stopper])

        train_prob = model.predict(train_val_tokens.tolist()).flatten()
        test_prob = model.predict(test_tokens).flatten()
        np.save(file=f"{out_folder}/train.npy", arr=train_prob, allow_pickle=True, fix_imports=True)
        np.save(file=f"{out_folder}/test.npy", arr=test_prob, allow_pickle=True, fix_imports=True)  

        test_pred = (test_prob > 0.5).astype(np.int)
        del model

    elif config.method =='RF':
        print("Random Forest model")
        model = RandomForestClassifier(n_estimators=100, verbose=1)
        model.fit(X=X_train, y=Y_train)
    elif config.method =='SVM':    
        model = SVC(kernel='rbf', cache_size=2000, probability=True)
        model.fit(X=X_train, y=Y_train)
    elif config.method =='LR':
        print("Logistic Regressinon")
        model =  LogisticRegression(random_state=config.random_seed, fit_intercept=False)
        model.fit(X=X_train, y=Y_train)
    elif config.method =='TREE':
        print("Decision Tree")
        model =  DecisionTreeClassifier(random_state=config.random_seed)
        model.fit(X=X_train, y=Y_train)
    elif config.method =='NB':
        print("Bernouli Naive Bayes")
        model =  BernoulliNB(alpha=1.0)
        model.fit(X=X_train, y=Y_train)
    elif config.method =='KN':
        print("K-Nearest neighbors")
        model =  KNeighborsClassifier(n_neighbors=8, leaf_size=10)
        model.fit(X=X_train, y=Y_train)
    else:
        raise ValueError("Unknown method")

    if config.method in ['RF', 'SVM', 'LR', 'TREE', 'NB', 'KN']:
        try:
            train_prob = model.predict_proba(X_train)[:,1]
            test_prob = model.predict_proba(X_test)[:,1]    
            np.save(file=f"{out_folder}/train.npy", arr=train_prob, allow_pickle=True, fix_imports=True)
            np.save(file=f"{out_folder}/test.npy", arr=test_prob, allow_pickle=True, fix_imports=True)            
        except:
            print("Cannot export the prediction prob")
        test_pred = model.predict(X_test)
        

    AVERAGE_USED = 'macro'
    test_accuracy = accuracy_score(1-Y_test, 1-test_pred)
    test_precision = precision_score(1-Y_test, 1-test_pred, average=AVERAGE_USED)
    test_recall = recall_score(1-Y_test, 1-test_pred, average=AVERAGE_USED)
    test_f1_score = f1_score(1-Y_test,1-test_pred, average=AVERAGE_USED)
    
    print(' Accuracy: %.3f' % (test_accuracy))
    print(' Precision: %.3f' % (test_precision))
    print(' Recall: %.3f' % (test_recall))
    print(' F1-Score: %.3f' % (test_f1_score))

  

if __name__ == '__main__':
    with open('./config/config_word2vec.yaml', 'r') as stream:
        config = yaml.safe_load(stream)
    config = edict(config)
    
    main(config)