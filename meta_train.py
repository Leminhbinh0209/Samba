import warnings
warnings.filterwarnings('ignore')
import os
import tensorflow as tf
from tensorflow.keras import backend as K

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from sklearn.model_selection import StratifiedKFold
try: 
    from cuml.svm import SVC # Cuda-supported SVM
except:
    from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import  DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.metrics import roc_curve, auc, accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import statistics
import json
import meta_helper.Dataset
from meta_helper.models import DISTURBED_YOUTUBE_MODEL, simple_dnn, simple_cnndnn, ENSEMBLE_DISTURBED_YOUTUBE_MODEL
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE
from tensorflow.keras.utils import to_categorical
import collections
import pickle
from time import time
from datetime import timedelta
from meta_helper.utils import *
from easydict import EasyDict as edict
import yaml
from meta_helper.h5py_func import read_dict

os.environ ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '6,7' 

with open('./config/config_meta.yaml', 'r') as stream:
        config = yaml.safe_load(stream)
config = edict(config)

print("Training with: ", config.method) 

def get_model_name(model_type_array):
    model_type_str = ''
    for i in range(0, len(model_type_array)):
        if i != len(model_type_array)-1:
            model_type_str += model_type_array[i] + '_'
        else:
            model_type_str += model_type_array[i]
    return model_type_str

def stratified_train_val_split(set_labels, val_size=0.2):
    # print('Splitting to TRAIN and VAL set. TOTAL Videos: %d' % (len(set_labels)))
    # Declare vaiables
    indices_train, indices_val = list(), list()

    # Get the indices of the Appropriate videos
    indices_appropriate = [i for i, x in enumerate(set_labels) if x == 0.0]

    # Get the indices of the Disturbing videos
    indices_disturbing = [i for i, x in enumerate(set_labels) if x == 1.0]

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

def train_test_model(k_fold=10,
                     apply_oversampling=True,
                     model_type=['all_inputs'],
                     other_features_type='statistics',
                     nb_classes=4,
                     nb_epochs=1,
                     batch_size=50,
                     validation_split=0.0,
                     shuffle_training_set=False,
                     dropout_level=0.0,
                     text_input_dropout_level=0.0,
                     learning_rate=1e-5,
                     adam_beta_1=0.9,
                     adam_beta_2=0.999,
                     decay=0.0,
                     epsilon=None,
                     loss_function='categorical_crossentropy',
                     early_stopping_patience=10,
                     filenames_extension=None,
                     final_dropout_level=0.5,
                     dimensionality_reduction_layers=True):
    """
    Model store absolute path with filename
    """

    MODELS_BASE_DIR = f"./checkpoints/{config.dataset.lower()}/classifier/training/" + get_model_name(model_type_array=model_type) + '_training/models/model' + filenames_extension
    if config.method == 'aaai':
        model_filename = 'disturbed_youtube_model_'
    # elif config.method == 'ensemble':
    #     model_filename = 'ensemble_model'
    elif config.method == 'dnn':
        model_filename = 'double_layer'
    elif config.method == 'cnn-dnn':
        model_filename = 'cnn_dnn_simplemodel'
    os.makedirs(MODELS_BASE_DIR, exist_ok = True) 
    print(MODELS_BASE_DIR)
    early_stopper = EarlyStopping(mode='auto', verbose=2, monitor='val_loss', restore_best_weights=True, patience=early_stopping_patience)

    ### Read and split dataset
    uscs_dir = f"{config.data_folder}/{config.dataset}_data/" 
    data_dir = f"{uscs_dir}/meta_data/meta_emb/"
    
        
    with open(data_dir + 'voc.json') as outfile:
        voc = json.load( outfile)
    headlines_vocab_size, headlines_words_seq_length = voc['headline']['voc_size'], voc['headline']['max']
    video_tags_vocab_size, video_tags_seq_length = voc['tag']['voc_size'], voc['tag']['max']
    
    print("Read embedding data...")
    instance_keys = ['thumbnail', 'headline', 'style', 'tags', 'y']
    inputs = read_dict(data_dir +  'meta_embedding.hdf5')

    all_video_id = [u.decode('UTF-8') for u in inputs['video_id'].tolist()]
    all_thumbnails_features = np.asarray(inputs[instance_keys[0]], dtype=np.float32)       
    all_headlines_features =  np.asarray(inputs[instance_keys[1]], dtype=np.float32)    
    all_statistics_features = np.asarray(inputs[instance_keys[2]], dtype=np.float32)    
    all_video_tags_features = np.asarray(inputs[instance_keys[3]], dtype=np.float32)    
    targets = inputs['y']
  
    assert all_headlines_features.shape[1] == headlines_words_seq_length, "Headline leng issue!!!"
    assert all_video_tags_features.shape[1] == video_tags_seq_length, "Tags length issue!!!"
    print("Number of samples : ", len(targets) ) 
    print("Number positive samples: ", len(targets) - np.sum(targets) )

    # Load training and test set
    
    """
    Train and Test Video id loading
    """
    f = open(f"{uscs_dir}{config.dataset.lower()}_train_videos.txt", "r")
    train_video_id = f.readlines()
    train_video_id = [i.strip() for i in train_video_id]
    f.close()

    f = open(f"{uscs_dir}{config.dataset.lower()}_test_videos.txt", "r")
    test_video_id = f.readlines()
    test_video_id = [i.strip() for i in test_video_id]
    f.close()

    index_lookup = dict(zip(all_video_id, np.arange(len(all_video_id))))
    train_val_set_indices = [index_lookup[u] for u in train_video_id]
    test_set_indices = [index_lookup[u] for u in test_video_id]

    # uscs_big_subtitle = pd.read_csv(uscs_dir + f"transcripts/{config.dataset.lower()}_big_subtitle.csv")
    # uscs_big_subtitle = uscs_big_subtitle.loc[uscs_big_subtitle['video_id'].isin(all_video_id)]
    # all_video_subtitle_features = uscs_big_subtitle.loc[:, 'subtitle'].values

    thumbnails_num_examples = len(all_thumbnails_features)

    dataset_labels = targets
    dataset_labels_binary = targets

    tic = time()
    
    # For final evaluation
    dataset_pred_binary = np.zeros_like(dataset_labels_binary)
    
    out_folder = f"{uscs_dir}/meta_data/{config.method}/"
    os.makedirs(out_folder, exist_ok=True)

    print('\n\n-----------------------------------------------------------------')
    print('--- [TRAIN: %d, TEST: %d' % (len(train_val_set_indices), len(test_set_indices)))
    # print('--------------------------------------------------------------------\n')

    """
    Get current Fold Video IDs and Labels for each Set
    """
    # TRAIN_VAL_video_ids = np.take(dataset_videos, train_set_indices, axis=0)
    Y_train_val = np.take(dataset_labels, indices=train_val_set_indices, axis=0)
    Y_train_val_binary = np.take(dataset_labels_binary, indices=train_val_set_indices, axis=0)

    # TEST_videos_ids = np.take(dataset_videos, test_set_indices, axis=0)
    Y_test = np.take(dataset_labels, indices=test_set_indices, axis=0)
    Y_test_binary = np.take(dataset_labels_binary, indices=test_set_indices, axis=0)

    # print('--- TEST_ID: %s, TEST_LABEL: %s' % (TEST_videos_ids[0], str(Y_test[0])))

    """
    Apply the Over-sampling technique on the TRAIN data using the SMOTE algorithm (https://arxiv.org/pdf/1106.1813.pdf)
    """
    if apply_oversampling:
        print("--- Apply upsampling ---")
        smote = SMOTE(sampling_strategy='all') # can be any of 'all', 'minority'
        dataset_class_weights = dict()
    else:
        dataset_class_weights = class_weight.compute_class_weight('balanced', np.unique(Y_test_binary), Y_test_binary)

    """
    Get TRAIN-TEST Input Data and Features for the current FOLD based on the indices of each set
    """
    """
    Get TRAIN-TEST Input Data and Features for the current FOLD based on the indices of each set
    """
    # Split TRAIN to TRAIN & VAL (basically get the indices)
    indices_train, indices_val = stratified_train_val_split(set_labels=Y_train_val_binary, val_size=validation_split)

    # Get Y_train and Y_val
    Y_train = np.take(Y_train_val, indices=indices_train, axis=0)
    Y_train_binary = np.take(Y_train_val_binary, indices=indices_train, axis=0)
    Y_val = np.take(Y_train_val, indices=indices_val, axis=0)
    Y_val_binary = np.take(Y_train_val_binary, indices=indices_val, axis=0)
    
    """
    THUMBNAILS
    """
    # TRAIN & VAL
    X_train_val_thumbnails = np.take(all_thumbnails_features, indices=train_val_set_indices, axis=0)
    X_train_thumbnails = np.take(X_train_val_thumbnails, indices=indices_train, axis=0)
    X_val_thumbnails = np.take(X_train_val_thumbnails, indices=indices_val, axis=0)

    # OVERSAMPLE TRAIN
    if apply_oversampling:
        X_train_thumbnails, Y_train_s = smote.fit_resample(X_train_thumbnails, Y_train_binary)

    # TEST
    X_test_thumbnails = np.take(all_thumbnails_features, indices=test_set_indices, axis=0)

    """
    HEADLINES
    """
    # TRAIN & VAL
    X_train_val_headlines = np.take(all_headlines_features, train_val_set_indices, axis=0)
    X_train_headlines = np.take(X_train_val_headlines, indices=indices_train, axis=0)
    X_val_headlines = np.take(X_train_val_headlines, indices=indices_val, axis=0)

    # OVERSAMPLE TRAIN
    if apply_oversampling:
        X_train_headlines, Y_train_s = smote.fit_resample(X_train_headlines, Y_train_binary)

    # TEST
    X_test_headlines = np.take(all_headlines_features, indices=test_set_indices, axis=0)

    """
    STATISTICS
    """
    # TRAIN & VAL
    X_train_val_statistics = np.take(all_statistics_features, train_val_set_indices, axis=0)
    X_train_statistics = np.take(X_train_val_statistics, indices=indices_train, axis=0)
    X_val_statistics = np.take(X_train_val_statistics, indices=indices_val, axis=0)

    # OVERSAMPLE TRAIN
    if apply_oversampling:
        X_train_statistics, Y_train_s = smote.fit_resample(X_train_statistics, Y_train_binary)

    # TEST
    X_test_statistics = np.take(all_statistics_features, indices=test_set_indices, axis=0)

    """
    VIDEO TAGS
    """
    # TRAIN & VAL
    X_train_val_video_tags = np.take(all_video_tags_features, indices=train_val_set_indices, axis=0)
    X_train_video_tags = np.take(X_train_val_video_tags, indices=indices_train, axis=0)
    X_val_video_tags = np.take(X_train_val_video_tags, indices=indices_val, axis=0)

    # OVERSAMPLE TRAIN
    if apply_oversampling:
        X_train_video_tags, Y_train_s = smote.fit_resample(X_train_video_tags, Y_train_binary)

    # TEST
    X_test_video_tags = np.take(all_video_tags_features, indices=test_set_indices, axis=0)

    """
    VIDEOS SUBTILE
    """
    # X_train_val_video_subtitle = np.take(all_video_subtitle_features, indices=train_val_set_indices, axis=0)
    # X_train_video_subtitle = np.take(X_train_val_video_subtitle, indices=indices_train, axis=0)
    # X_val_video_subtitle = np.take(X_train_val_video_subtitle, indices=indices_val, axis=0)

    # OVERSAMPLE TRAIN
    # if apply_oversampling:
    #     X_train_video_subtitle, Y_train_s = smote.fit_resample(X_train_video_subtitle, Y_train_binary)

    # TEST
    # X_test_video_subtitle = np.take(all_video_subtitle_features, indices=test_set_indices, axis=0) 

    # Get Oversampled categorical Y_train
    if apply_oversampling:
        Y_train_oversampled = np.array([to_categorical(label, nb_classes) for label in Y_train_s])
        # print('Checking if everything is OK: %s' % (str(np.array_equal(X_train_val_video_tags, np.concatenate((X_train_video_tags, X_val_video_tags), axis=0)))))
        print('--- [AFTER OVER-SAMPLING] TRAIN: %d, VAL: %d, TEST: %d' % (Y_train_oversampled.shape[0], Y_val.shape[0], Y_test.shape[0]))
        if loss_function == 'binary_crossentropy':
            Y_train_oversampled = Y_train_s
    else:
        Y_train_oversampled = np.array([to_categorical(label, nb_classes) for label in Y_train]) 
        if loss_function == 'binary_crossentropy':
            Y_train_oversampled = Y_train
    print('--------------------------------------------------------------------\n')
    """
    Examine Class Distribution
    """
    print('Y_TRAIN: %s' % (str(collections.Counter(Y_train_binary))))
    if apply_oversampling:
        print('Y_TRAIN_OVERSAMPLED: %s' % (str(collections.Counter(Y_train_s))))
    if validation_split > 0.0:
        print('Y_VAL: %s' % (str(collections.Counter(Y_val_binary))))
    print('Y_TEST: %s' % (str(collections.Counter(Y_test_binary))))
    
    """
    Create Current Model's Directory
    """
    # Check if model directory exists
    original_umask = os.umask(0)
    try:
        # Create Thumbnails Base Directory
        if not os.path.exists(MODELS_BASE_DIR ):
            os.makedirs(MODELS_BASE_DIR , 0o777)
    finally:
        os.umask(original_umask)


    """
    [TRAIN] Fit the model to start training
    """
    print('--- Started TRAINing the Model')
    print("Shape of Thumbnails", X_train_thumbnails.shape)
    # print("Subtitle length:", len(X_train_video_subtitle))
    # SET MODEL INPUTS
    model_input, model_val_input = list(), list()
    model_input.append(X_train_thumbnails)
    model_input.append(X_train_headlines)
    model_input.append(X_train_statistics)
    model_input.append(X_train_video_tags)
    config.input_dim = X_train_thumbnails.shape[1] + X_train_headlines.shape[1] + X_train_statistics.shape[1] + X_train_video_tags.shape[1]

    model_val_input.append(X_val_thumbnails)
    model_val_input.append(X_val_headlines)
    model_val_input.append(X_val_statistics)
    model_val_input.append(X_val_video_tags)
    Y_val_onehot = np.array([to_categorical(label, nb_classes) for label in Y_val])
    # if loss_function == 'binary_crossentropy':
    #         Y_val_onehot = Y_val
    print('='*10 + " Start fitting model " +'='*10 )
    if config.method == 'aaai':
        print("AAAI proposed methods")

        disturbed_youtube_model = DISTURBED_YOUTUBE_MODEL(saved_model_path=None,
                                                thumbnails_num_examples=thumbnails_num_examples,
                                                headlines_words_seq_length=headlines_words_seq_length,
                                                headlines_vocab_size=headlines_vocab_size,
                                                video_tags_seq_length=video_tags_seq_length,
                                                video_tags_vocab_size=video_tags_vocab_size,
                                                other_features_type=other_features_type,
                                                nb_classes=nb_classes,
                                                nb_epochs=nb_epochs,
                                                dropout_level=dropout_level,
                                                text_input_dropout_level=text_input_dropout_level,
                                                batch_size=batch_size,
                                                learning_rate=learning_rate,
                                                adam_beta_1=adam_beta_1,
                                                adam_beta_2=adam_beta_2,
                                                decay=decay,
                                                epsilon=epsilon,
                                                loss_function=loss_function,
                                                final_dropout_level=final_dropout_level,
                                                dimensionality_reduction_layers=dimensionality_reduction_layers)
        print(batch_size, shuffle_training_set)
        disturbed_youtube_model.model.fit(model_input,
                                        Y_train_oversampled,
                                        batch_size=batch_size,
                                        validation_data=(model_val_input, Y_val_onehot),
                                        shuffle=shuffle_training_set,
                                        verbose=1,
                                        # callbacks=[ early_stopper],#
                                        epochs=nb_epochs)
        """
        SAVE the Model
        """
        print('\n--- TRAINing has finished. SAVING THE FULL MODEL...')
        final_model_store_path = MODELS_BASE_DIR + '/' + model_filename + 'final.tf'
        os.makedirs(MODELS_BASE_DIR  + '/' , exist_ok=True)
        # Save the whole Model with its weights
        disturbed_youtube_model.model.save(final_model_store_path, save_format='tf')

    # if config.method == 'ensemble':
    #     print("Proposed Ensemblem method")
    #     model_input.append(X_train_video_subtitle)
    #     model_val_input.append(X_val_video_subtitle)
    #     disturbed_youtube_model = ENSEMBLE_DISTURBED_YOUTUBE_MODEL(saved_model_path=None,
    #                                             thumbnails_num_examples=thumbnails_num_examples,
    #                                             headlines_words_seq_length=headlines_words_seq_length,
    #                                             headlines_vocab_size=headlines_vocab_size,
    #                                             video_tags_seq_length=video_tags_seq_length,
    #                                             video_tags_vocab_size=video_tags_vocab_size,
    #                                             train_subtitle=X_train_video_subtitle,
    #                                             other_features_type=other_features_type,
    #                                             nb_classes=nb_classes,
    #                                             nb_epochs=nb_epochs,
    #                                             dropout_level=dropout_level,
    #                                             text_input_dropout_level=text_input_dropout_level,
    #                                             batch_size=batch_size,
    #                                             learning_rate=learning_rate,
    #                                             adam_beta_1=adam_beta_1,
    #                                             adam_beta_2=adam_beta_2,
    #                                             decay=decay,
    #                                             epsilon=epsilon,
    #                                             loss_function=loss_function,
    #                                             final_dropout_level=final_dropout_level,
    #                                             dimensionality_reduction_layers=dimensionality_reduction_layers)
        
    #     disturbed_youtube_model.model.fit(model_input,
    #                                     Y_train_oversampled,
    #                                     batch_size=batch_size,
    #                                     validation_data=(model_val_input, Y_val_onehot),
    #                                     shuffle=shuffle_training_set,
    #                                     verbose=1,
    #                                     callbacks=[early_stopper ],
    #                                     epochs=nb_epochs) # 
    #     """
    #     SAVE the Model
    #     """
    #     print('\n--- TRAINing has finished. SAVING THE FULL MODEL...')
    #     final_model_store_path = MODELS_BASE_DIR + '/' + str(kfold_cntr) + '/' + model_filename + 'final.tf'
    #     # Save the whole Model with its weights
    #     disturbed_youtube_model.model.save(final_model_store_path, save_format='tf')

    elif config.method == 'dnn':
        print("Double layer deep neural networks")
        model = simple_dnn(config.input_dim, None)
        model.fit(np.column_stack(model_input),
                Y_train_oversampled,
                batch_size=batch_size,
                validation_data=(np.column_stack(model_val_input), Y_val_onehot),
                shuffle=shuffle_training_set,
                verbose=1,
                # callbacks=[early_stopper ],
                epochs=nb_epochs) 

        print('\n--- TRAINing has finished. SAVING THE FULL MODEL...')
        final_model_store_path = MODELS_BASE_DIR + '/' + model_filename + 'final.tf'
        # Save the whole Model with its weights
        model.save(final_model_store_path)
        # Delete the Model
        del model
    elif config.method == 'cnn-dnn':
        print("Double layer deep neural networks")
        model = simple_cnndnn(config.input_dim, None)
        model.fit(np.expand_dims(np.column_stack(model_input), axis=-1),
                Y_train_oversampled,
                batch_size=batch_size,
                validation_data=(np.expand_dims(np.column_stack(model_val_input), axis=-1), Y_val_onehot),
                shuffle=shuffle_training_set,
                verbose=1,
                # callbacks=[early_stopper ],
                epochs=nb_epochs) 

        print('\n--- TRAINing has finished. SAVING THE FULL MODEL...')
        final_model_store_path = MODELS_BASE_DIR + '/' + model_filename + 'final.tf'
        # Save the whole Model with its weights
        model.save(final_model_store_path)
        # Delete the Model
        del model

    elif config.method =='RF':
        print("Random Forest model")
        model = RandomForestClassifier(n_estimators=100, verbose=1, criterion='entropy')
        model.fit(X=np.column_stack(model_input), y=Y_train_oversampled.argmax(axis=1))
    elif config.method =='SVM':    
        
        model = SVC(kernel='rbf', C=10, gamma=1, cache_size=2000)
        model.fit(X=np.column_stack(model_input), y=Y_train_oversampled.argmax(axis=1))
    elif config.method =='LR':
        print("Logistic Regressinon")
        model =  LogisticRegression(random_state=config.random_seed)
        model.fit(X=np.column_stack(model_input), y=Y_train_oversampled.argmax(axis=1))
    elif config.method =='TREE':
        print("Decision Tree")
        model =  DecisionTreeClassifier(random_state=config.random_seed,  criterion='entropy')
        model.fit(X=np.column_stack(model_input), y=Y_train_oversampled.argmax(axis=1))
    elif config.method =='NB':
        print("Bernouli Naive Bayes")
        model =  BernoulliNB(alpha=1.0)
        model.fit(X=np.column_stack(model_input), y=Y_train_oversampled.argmax(axis=1))
    elif config.method =='KN':
        print("K-Nearest neighbors")
        model =  KNeighborsClassifier(n_neighbors=8, leaf_size=10)
        model.fit(X=np.column_stack(model_input), y=Y_train_oversampled.argmax(axis=1))
        
    else:
        raise ValueError(f"Unknown method {config.method}")
    """
    [TEST] Evaluate the model 
    """
    # SET MODEL INPUTS
    model_test_input = [X_test_thumbnails, X_test_headlines, X_test_statistics, X_test_video_tags]
    
    if config.method in ['RF', 'SVM', 'LR', 'TREE', 'NB', 'KN']:
        test_pred = model.predict(np.column_stack(model_test_input))

    elif config.method == 'aaai':
        saved_model_path = MODELS_BASE_DIR + '/' + model_filename + 'final.tf'
        disturbed_youtube_model = DISTURBED_YOUTUBE_MODEL(load_saved_model=True, saved_model_path=saved_model_path)
        print('Model Loaded successfully from directory!')
        test_pred_prob = disturbed_youtube_model.model.predict(model_test_input, batch_size=batch_size, verbose=1, steps=None)
        np.save(file=f"{out_folder}/test.npy", arr=test_pred_prob, allow_pickle=True, fix_imports=True)  
        test_pred = test_pred_prob.argmax(axis=1)

        # elif config.method == 'ensemble':
        #     model_test_input.append(X_test_video_subtitle)
        #     # saved_model_path = MODELS_BASE_DIR + '/' + str(kfold_cntr) + '/' + model_filename + 'final.tf'
        #     # disturbed_youtube_model = ENSEMBLE_DISTURBED_YOUTUBE_MODEL(load_saved_model=True, 
        #     #                                                     saved_model_path=saved_model_path,
        #     #                                                     thumbnails_num_examples=thumbnails_num_examples,
        #     #                                                     headlines_words_seq_length=headlines_words_seq_length,
        #     #                                                     headlines_vocab_size=headlines_vocab_size,
        #     #                                                     video_tags_seq_length=video_tags_seq_length,
        #     #                                                     video_tags_vocab_size=video_tags_vocab_size,
        #     #                                                     train_subtitle=X_train_video_subtitle,
        #     #                                                     other_features_type=other_features_type,
        #     #                                                     nb_classes=nb_classes,
        #     #                                                     nb_epochs=nb_epochs,
        #     #                                                     dropout_level=dropout_level,
        #     #                                                     text_input_dropout_level=text_input_dropout_level,
        #     #                                                     batch_size=batch_size,
        #     #                                                     learning_rate=learning_rate,
        #     #                                                     adam_beta_1=adam_beta_1,
        #     #                                                     adam_beta_2=adam_beta_2,
        #     #                                                     decay=decay,
        #     #                                                     epsilon=epsilon,
        #     #                                                     loss_function=loss_function,
        #     #                                                     final_dropout_level=final_dropout_level,
        #     #                                                     dimensionality_reduction_layers=dimensionality_reduction_layers)

        #     # print('--- Model Loaded successfully from directory!')
        #     test_pred_proba = disturbed_youtube_model.model.predict(model_test_input, batch_size=batch_size, verbose=1, steps=None)
        #     if loss_function == 'binary_crossentropy':
        #         test_pred = (test_pred_proba > 0.5).astype(np.int16)
        #     else:
        #         test_pred = test_pred_proba.argmax(axis=1)

    elif config.method in ['dnn']:
        saved_model_path = MODELS_BASE_DIR + '/'  + model_filename + 'final.tf'
        dnn_model = simple_dnn(config.input_dim, saved_model_path)
        print('--- Model Loaded successfully from directory!')
        test_pred_proba = dnn_model.predict(np.column_stack(model_test_input), batch_size=batch_size, verbose=1, steps=None)
        if loss_function == 'binary_crossentropy':
            test_pred = (test_pred_proba > 0.5).astype(np.int16)
        else:
            test_pred = test_pred_proba.argmax(axis=1)
    

    elif config.method in ['cnn-dnn']:
        saved_model_path = MODELS_BASE_DIR + '/' + model_filename + 'final.tf'
        dnn_model = simple_cnndnn(config.input_dim, saved_model_path)
        print('--- Model Loaded successfully from directory!')
        test_pred_proba = dnn_model.predict(np.expand_dims(np.column_stack(model_test_input), axis=-1), batch_size=batch_size, verbose=1, steps=None)
        if loss_function == 'binary_crossentropy':
            test_pred = (test_pred_proba > 0.5).astype(np.int16)
        else:
            test_pred = test_pred_proba.argmax(axis=1)

    dataset_pred_binary[test_set_indices] = test_pred
    AVERAGE_USED = 'macro'
    test_accuracy = accuracy_score(1-Y_test_binary, 1-test_pred)
    test_precision = precision_score(1-Y_test_binary, 1-test_pred, average=AVERAGE_USED)
    test_recall = recall_score(1-Y_test_binary, 1-test_pred, average=AVERAGE_USED)
    test_f1_score = f1_score(1-Y_test_binary,1-test_pred, average=AVERAGE_USED)
    
    print('\033[92m--- TEST Accuracy: %.3f' % (test_accuracy))
    print('--- TEST Precision: %.3f' % (test_precision))
    print('--- TEST Recall: %.3f' % (test_recall))
    print('--- TEST F1-Score: %.3f' % (test_f1_score))
    print('\033[0m')
  

if __name__ == '__main__':
    filenames_extension = '_id=' + str(config.RANDOM_ID) + \
                        '_K=' + str(config.k_fold) + \
                        '_nbclasses=' + str(config.NB_CLASSES) + \
                        '_epoch=' + str(config.NB_EPOCHS) + \
                        '_drop=' + str(config.dropout_level) + \
                        '_opt=' + config.optimizer_type + \
                        '_lr=' + str(config.learning_rate) + \
                        '_eps=' + str(config.epsilon) + \
                        '_fdrop=' + str(config.include_final_dropout_layer)

    """
    Call the train_test_model function to start TRAIN and also TEST the Disturbed YouTube Model
    """

    train_test_model(k_fold=config.k_fold,
                    apply_oversampling=config.apply_oversampling,
                    model_type=['all_input'],
                    other_features_type=config.other_features_type,
                    nb_classes=config.NB_CLASSES,
                    nb_epochs=config.NB_EPOCHS,
                    batch_size=config.BATCH_SIZE,
                    validation_split=config.validation_split,
                    shuffle_training_set=config.shuffle_training_set,
                    dropout_level=config.dropout_level,
                    text_input_dropout_level=config.text_input_dropout_level,
                    learning_rate=config.learning_rate,
                    epsilon=config.epsilon,
                    loss_function=config.loss_function,
                    early_stopping_patience=config.early_stopping_patience,
                    filenames_extension=filenames_extension,
                    final_dropout_level=config.final_dropout_level,
                    dimensionality_reduction_layers=True)