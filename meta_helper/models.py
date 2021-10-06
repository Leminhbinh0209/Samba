# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 20:33:33 2018

@author: kostantinos.papadamou

This class implements the Deep Learning Model for the detection of disturbing videos on YouTube.
The data_peraparion.py uses a DATASET class which implements all the required methods for the retrieval and
pre-processing of all the data that we as input to our model.

Below we describe the architecture of our DL Model:
    1. [THUMBNAILS] --> CNN (pre-trained Inception V3) ------------------------->  |
    2. [TITLES (HEADLINES)] --> EMBEDDING --> LSTM ----------------------------->  |  --> FUSING (DNN) --> SOFTMAX
    3. [STATISTICS] --> DNN ---------------------------------------------------->  |
    4. [TAGS] --> EMBEDDING --> LSTM ------------------------------------------->  |

NOTE: For both the VIDEO FRAMES and the THUMBNAILS we have generated from before their features by passing them
to the pre-trained CNN Inception V3 model.
"""

import os
import tensorflow as tf

"""
GENERAL CONFIGURATION
"""
# Restrict the script to run everything on the first GPU [GPU_ID=0]
os.environ ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Solve a bug with Tensorflow v0.11
# tf.python.control_flow_ops = tf

# Import Keras Tensoflow Backend
from tensorflow.keras import backend as K
from collections import deque
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate, Lambda, Reshape, Convolution2D, MaxPooling2D, \
    GlobalAveragePooling2D, Conv1D, GlobalAveragePooling1D, MaxPooling1D, Flatten, GlobalMaxPooling1D, Average
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
# from tensorflow.keras import objectives
from tensorflow.keras.applications.inception_v3 import InceptionV3
import string
import re
# Calculate Precision (Batch-wise)
def PRECISION(y_true, y_pred):
    """Precision metric.
        Only computes a batch-wise average of precision.
        Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

# Calculate Recall (Batch-wise)
def RECALL(y_true, y_pred):
    """Recall metric.
        Only computes a batch-wise average of recall.
        Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def F1_SCORE(y_true, y_pred):
    """Computes the F1 Score
    Only computes a batch-wise average of recall. Computes the recall, a metric
    for multi-label classification of how many relevant items are selected.
    """
    p = PRECISION(y_true, y_pred)
    r = RECALL(y_true, y_pred)
    return (2 * p * r) / (p + r + K.epsilon())


class DISTURBED_YOUTUBE_MODEL(object):
    def __init__(self,
                 load_saved_model=False,
                 saved_model_path=None,
                 thumbnails_num_examples=0,
                 headlines_words_seq_length=0,
                 headlines_vocab_size=0,
                 video_tags_seq_length=0,
                 video_tags_vocab_size=0,
                 other_features_type='all',
                 nb_classes=4,
                 nb_epochs=1000,
                 dropout_level=0.0,
                 text_input_dropout_level=0.0,
                 batch_size=50,
                 learning_rate=1e-5,
                 adam_beta_1=0.9,
                 adam_beta_2=0.999,
                 decay=0.0,
                 epsilon=None,
                 loss_function='categorical_crossentropy',
                 final_dropout_level=0.5,
                 dimensionality_reduction_layers=True):

        """
        Init Model-related parameters
        """
        self.MODELS_BASE_DIR = './classifier/training/'
        self._saved_model_path = saved_model_path

        """
        Init other parameters
        """
        self._thumbnails_num_examples = thumbnails_num_examples
        self._headlines_words_seq_length = headlines_words_seq_length # the max number of words that we have in a headline
        self._headlines_vocab_size = headlines_vocab_size
        self._video_tags_seq_length = video_tags_seq_length
        self._video_tags_vocab_size = video_tags_vocab_size
        self._other_features_type = other_features_type
        self.nb_classes = nb_classes
        self.nb_epochs = nb_epochs
        self.dropout_level = dropout_level
        self.text_input_dropout_level = text_input_dropout_level
        self.batch_size = batch_size

        """
        Implement custom Batch-wise Metrics (Precision, Recall, F1 Score, and AUC Score)
        """
        

        # Calculate AUC Score (Batch-wise)
        def AUC_SCORE(y_true, y_pred):
            return 1
#             auc = tf.metrics.auc(y_true, y_pred, curve='ROC')[1]
#             K.get_session().run(tf.local_variables_initializer())
#             return auc


        """
        Define the metrics that are to be calculated during training
        """
        self.metrics = ['categorical_accuracy', PRECISION, RECALL, F1_SCORE, tf.keras.metrics.AUC(name='auc')]

        """
        Set OPTIMIZATION and other model parameters
        """
        # Set Optimization parameters
        self.learning_rate = learning_rate
        self.adam_beta_1 = adam_beta_1
        self.adam_beta_2 = adam_beta_2
        self.decay = decay
        self.epsilon = epsilon
        # Set other model parameters
        self.loss_function = loss_function

        """
        Set OTHER model parameters
        """
        self.final_dropout_level = final_dropout_level
        self.dimensionality_reduction_layers = dimensionality_reduction_layers

        """
        Build and compile the Full Model
        """
        if load_saved_model:
            # Load an already SAVED Model

            print('Loading saved model from %s ...' % self._saved_model_path)
     
            self.model = load_model(filepath=self._saved_model_path,
                                    custom_objects={
                                        'PRECISION': PRECISION,
                                        'RECALL': RECALL,
                                        'F1_SCORE': F1_SCORE,
                                        'AUC_SCORE':  tf.keras.metrics.AUC(name='auc')
                                    })
        else:
            # Build the Full Model with all the features as Input
            self.model = self.build_all_inputs_full_model()

    """
    Model that accepts as input the 2048-d (1x2048) thumbnail features vector that we have extracted from our 
    pre-trained Inception V3 CNN. It includes a Dense layer that will accept the extracted features and will 
    then pass them to the FUSING Network during the Merge operation.
    
    Output: 2048-d vector
    """
    @staticmethod
    def thumbnail_submodel(model_structure_type='functional'):
        # declare input shape
        input_shape = (2048,)

        if model_structure_type == 'functional':
            # Declare Thumbnail Input
            thumbnail_in = Input(shape=input_shape, name='thumbnail_input')

            return thumbnail_in
        elif model_structure_type == 'sequential':
            # Create Model
            model = Sequential()

            # Declare Thumbnail Input
            model.add(Input(shape=input_shape, name='thumbnail_input'))
            return model

    """
    Model that accepts as input the thumbnail contents vector and fine-tunes an Inception V3 CNN. 

    Output: 2048-d vector
    """
    @staticmethod
    def thumbnail_CNN_submodel():
        # Add the CNN Inception V3 model
        cnn = InceptionV3(weights='imagenet', include_top=False)

        # get layers and add average pooling layer
        cnn_out = cnn.output

        # Add an Average Pooling Layer
        average_pool_out = GlobalAveragePooling2D(name='average_pooling_layer')(cnn_out)

        model = Model(inputs=cnn.input, outputs=average_pool_out)

        # Freeze pre-trained CNN model area's layer
        for layer in cnn.layers:
            layer.trainable = False

        return model

    """
    Model that accepts as input the HEADLINES FEATURES of size (SET_LENGTH x MAX_WORDS_IN HEADLINE) and  
    processes the HEADLINE in order the extract useful representations from the words in it. 
    It includes and Embedding layer and an LSTM that accepts as a sequence each word of the headline.
    
    Output: 100-d vector
    """
    def headline_submodel(self):
        # Init Embedding Layer variables
        _vocab_size = self._headlines_vocab_size
        # _vocab_size = 1000 # equivalent to the max_features parameter of the CountVectorizer in data_preparation
        embedding_vector_length = 32

        # Init LSTM variables
        lstm_headline_units = embedding_vector_length
        lstm_headline_input_shape = (self._headlines_words_seq_length, embedding_vector_length)

        # Create Model
        model = Sequential()
        # Add Embedding Layer
        model.add(Embedding(_vocab_size + 1,
                            embedding_vector_length,
                            input_length=self._headlines_words_seq_length,
                            trainable=True,
                            name='headline_embedding_layer')) # TRUE: update weights during training

        # Add LSTM
        model.add(LSTM(lstm_headline_units,
                       activation='tanh',
                       kernel_initializer='glorot_uniform', # 'uniform' or 'glorot_uniform'
                       go_backwards=False,
                       return_sequences=False,
                       input_shape=lstm_headline_input_shape,
                       dropout=self.text_input_dropout_level,
                       recurrent_dropout=0.0,
                       name='headline_lstm'))

        return model

    """
    Model that accepts as input the preprocessed statistics of each video of size (1x5) and uses a dense layer to 
    extract a feature representation from them that will be then passed to the FUSING Network.
    
    Output 1: 5-d vector for statistics features only
    Output 2: 25-d vector for all style features    
    """
    def statistics_submodel(self):
        # Init variables
        if self._other_features_type == 'statistics':
            dense_statistics_units = 5
            dense_dtype = 'float32'
        elif self._other_features_type == 'all':
            dense_statistics_units = 25
            dense_dtype = 'float32'

        statistics_dense_input_shape = (dense_statistics_units,)

        # Create Model
        model = Sequential()
        #Add a Dense Layer
        model.add(Dense(dense_statistics_units,
                        input_shape=statistics_dense_input_shape,
                        activation='relu',
                        kernel_initializer='glorot_uniform', # 'uniform' or 'glorot_uniform'
                        dtype=dense_dtype,
                        name='statistics_fully_connected'))
        return model

    """
    Model that accepts as input the VIDEO TAGS FEATURES of size (SET_LENGTH x MAX_TAGS_IN_A_VIDEO)  
    and processes the TAGS in order to extract useful representations between them in a video. 
    It includes and Embedding layer and an LSTM that accepts as a sequence each video tag.

    Output: 100-d vector
    """
    def video_tags_submodel(self):
        # Init Embedding Layer variables
        _vocab_size = self._video_tags_vocab_size
        embedding_vector_length = 32

        # Init LSTM variables
        lstm_video_tags_units = embedding_vector_length
        lstm_video_tags_input_shape = (self._video_tags_seq_length, embedding_vector_length)

        # Create Model
        model = Sequential()

        # Add Embedding Layer
        model.add(Embedding(_vocab_size + 1,
                            embedding_vector_length,
                            input_length=self._video_tags_seq_length,
                            trainable=True,
                            name='video_tags_embedding_layer'))

        # Add LSTM
        model.add(LSTM(lstm_video_tags_units,
                       activation='tanh',
                       kernel_initializer='glorot_uniform',
                       # go_backwards=False,
                       return_sequences=False,
                       input_shape=lstm_video_tags_input_shape,
                       dropout=self.text_input_dropout_level,
                       recurrent_dropout=0.0,
                       name='video_tags_lstm'))
        return model

    """
    Method that takes ALL different sub-models and builds the whole model. It also splits the input to the 
    different parts required to be fed to each submodel. At the end it concats all the outputs of the sub-models
    and pass that to a Fully connected layer and then to a Softmax layer that performs the classification.
    """
    def build_all_inputs_full_model(self):
        print('\n-------------------------------------------------')
        print('---      Started building the FULL MODEL      ---')
        print('-------------------------------------------------\n')

        """
        Get the THUMBNAIL branch model
        """
        thumbnail_branch = self.thumbnail_submodel(model_structure_type='sequential')
        # thumbnail_branch = self.thumbnail_CNN_submodel()

        """
        Get the HEADING (TITLE) branch model
        """
        heading_branch = self.headline_submodel()

        """
        Get the STATISTICS branch model
        """
        statistics_branch = self.statistics_submodel()

        """
        Get the VIDEO TAGS branch model
        """
        video_tags_branch = self.video_tags_submodel()

        """
        Merge together all Branches
        """
        merged_inputs = Concatenate(name='x_merged_inputs')([thumbnail_branch.output, heading_branch.output, statistics_branch.output, video_tags_branch.output])

        """
        Add multiple Fully-Connected DENSE Layers for Dimensionality Reduction
        """
        fully_connected_3 = Dense(512, activation='relu')(merged_inputs)

        """
        Add a DROPOUT layer before the classification layer
        """
        if self.dimensionality_reduction_layers:
            dropout_layer = Dropout(self.final_dropout_level)(fully_connected_3)
        else:
            # dropout_layer = Dropout(self.final_dropout_level)(fully_connected_0)
            dropout_layer = Dropout(self.final_dropout_level)(merged_inputs)

        # Add a SOFTMAX layer at the end to perform the classification
        all_inputs_out = Dense(self.nb_classes,
                               activation='softmax',
                               name='classification_layer')(dropout_layer)

        """
        Create the Functional Model and declare Input and Output
        """
        model_inputs = list()
        model_inputs.append(thumbnail_branch.input)
        model_inputs.append(heading_branch.input)
        model_inputs.append(statistics_branch.input)
        model_inputs.append(video_tags_branch.input)
        full_model = Model(model_inputs, all_inputs_out)

        """
        Choose an Optimizer
        """
        # Declare an Optimizer
        optimizer = Adam(learning_rate=self.learning_rate,
                         beta_1=self.adam_beta_1,
                         beta_2=self.adam_beta_2,
                         decay=self.decay,  # learning rate decay over each update.
                         epsilon=self.epsilon,  # Fuzz factor. If None, defaults to K.epsilon()
                         amsgrad=False)  # whether to apply AMSGrad variant of Adam algorithm or not

        """
        Compile the Model
        """
        full_model.compile(loss=self.loss_function, optimizer=optimizer, metrics=self.metrics)

        """
        Summarize the Model
        """
        print(full_model.summary())
        return full_model

def simple_dnn(input_dim=2160, model_path=None):
    """
    A simple deep eural network 2 layer
    """
    # define the keras model
    model = Sequential()
    model.add(Dense(512, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax', name='classification_layer'))
    if model_path is not None:
        model = load_model(filepath=model_path, 
                custom_objects={
                                'PRECISION': PRECISION,
                                'RECALL': RECALL,
                                'F1_SCORE': F1_SCORE,
                                'AUC_SCORE':  tf.keras.metrics.AUC(name='auc')
                            })
        return model 
    # Declare an Optimizer
    learning_rate = 1e-3 # Usually between: 1e-3 (0.001) ... 1e-5 (0.00001)
    epsilon = 1e-4 # None, 1e-4...1e-8 | when training an Inception Network ideal values are 1.0 and 0.1 (default=1e-8)
    adam_beta_1 = 0.9
    adam_beta_2 = 0.999
    decay = 0.0 # learning rate decay
    optimizer = Adam(learning_rate=learning_rate,
                        beta_1=adam_beta_1,
                        beta_2=adam_beta_2,
                        decay=decay,  # learning rate decay over each update.
                        epsilon=epsilon,  # Fuzz factor. If None, defaults to K.epsilon()
                        amsgrad=False)  # whether to apply AMSGrad variant of Adam algorithm or not

    model.compile(loss='categorical_crossentropy', 
                optimizer=optimizer, 
                metrics=['categorical_accuracy', PRECISION, RECALL, F1_SCORE, tf.keras.metrics.AUC(name='auc')])
    return model

def simple_cnndnn(input_dim=2160, model_path=None):
    """
    Simple CNN- DNN
    """
    # define the keras model
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=7, strides=3, input_shape=(input_dim, 1))) #, input_shape=(input_dim, 1)
    model.add(GlobalAveragePooling1D())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax', name='classification_layer'))
    if model_path is not None:
        model = load_model(filepath=model_path, 
                custom_objects={
                                'PRECISION': PRECISION,
                                'RECALL': RECALL,
                                'F1_SCORE': F1_SCORE,
                                'AUC_SCORE':  tf.keras.metrics.AUC(name='auc')
                            })
        return model 
    # Declare an Optimizer
    learning_rate = 1e-3 # Usually between: 1e-3 (0.001) ... 1e-5 (0.00001)
    epsilon = 1e-4 # None, 1e-4...1e-8 | when training an Inception Network ideal values are 1.0 and 0.1 (default=1e-8)
    adam_beta_1 = 0.9
    adam_beta_2 = 0.999
    decay = 0.0 # learning rate decay
    optimizer = Adam(learning_rate=learning_rate,
                        beta_1=adam_beta_1,
                        beta_2=adam_beta_2,
                        decay=decay,  # learning rate decay over each update.
                        epsilon=epsilon,  # Fuzz factor. If None, defaults to K.epsilon()
                        amsgrad=False)  # whether to apply AMSGrad variant of Adam algorithm or not

    model.compile(loss='categorical_crossentropy', 
                optimizer=optimizer, 
                metrics=['categorical_accuracy', PRECISION, RECALL, F1_SCORE, tf.keras.metrics.AUC(name='auc')])
    return model

class ENSEMBLE_DISTURBED_YOUTUBE_MODEL(object):
    def __init__(self,
                 load_saved_model=False,
                 saved_model_path=None,
                 thumbnails_num_examples=0,
                 headlines_words_seq_length=0,
                 headlines_vocab_size=0,
                 video_tags_seq_length=0,
                 video_tags_vocab_size=0,
                 train_subtitle=None,
                 other_features_type='all',
                 nb_classes=4,
                 nb_epochs=1000,
                 dropout_level=0.0,
                 text_input_dropout_level=0.0,
                 batch_size=50,
                 learning_rate=1e-5,
                 adam_beta_1=0.9,
                 adam_beta_2=0.999,
                 decay=0.0,
                 epsilon=None,
                 loss_function='categorical_crossentropy',
                 final_dropout_level=0.5,
                 dimensionality_reduction_layers=True):

        """
        Init Model-related parameters
        """
        self.MODELS_BASE_DIR = './classifier/training/'
        self._saved_model_path = saved_model_path

        """
        Init other parameters
        """
        self._thumbnails_num_examples = thumbnails_num_examples
        self._headlines_words_seq_length = headlines_words_seq_length # the max number of words that we have in a headline
        self._headlines_vocab_size = headlines_vocab_size
        self._video_tags_seq_length = video_tags_seq_length
        self._video_tags_vocab_size = video_tags_vocab_size
        self._train_tokens = train_subtitle

        self._other_features_type = other_features_type
        self.nb_classes = nb_classes
        self.nb_epochs = nb_epochs
        self.dropout_level = dropout_level
        self.text_input_dropout_level = text_input_dropout_level
        self.batch_size = batch_size

        """
        Implement custom Batch-wise Metrics (Precision, Recall, F1 Score, and AUC Score)
        """
        """
        Define the metrics that are to be calculated during training
        """
        self.metrics = ['categorical_accuracy', PRECISION, RECALL, F1_SCORE, tf.keras.metrics.AUC(name='auc')]

        """
        Set OPTIMIZATION and other model parameters
        """
        # Set Optimization parameters
        self.learning_rate = learning_rate
        self.adam_beta_1 = adam_beta_1
        self.adam_beta_2 = adam_beta_2
        self.decay = decay
        self.epsilon = epsilon
        # Set other model parameters
        self.loss_function = loss_function

        """
        Set OTHER model parameters
        """
        self.final_dropout_level = final_dropout_level
        self.dimensionality_reduction_layers = dimensionality_reduction_layers

        """
        Build and compile the Full Model
        """
        if load_saved_model:
            # Load an already SAVED Model
            print('Loading saved model from %s ...' % self._saved_model_path)
            self.model = self.build_all_inputs_full_model()
            self.model.load_weights(self._saved_model_path)
            # self.model = load_model(filepath=self._saved_model_path,
            #                         custom_objects={
            #                             'PRECISION': PRECISION,
            #                             'RECALL': RECALL,
            #                             'F1_SCORE': F1_SCORE,
            #                             'AUC_SCORE':  tf.keras.metrics.AUC(name='auc')
            #                         })
        else:
            # Build the Full Model with all the features as Input
            self.model = self.build_all_inputs_full_model()

    """
    Model that accepts as input the 2048-d (1x2048) thumbnail features vector that we have extracted from our 
    pre-trained Inception V3 CNN. It includes a Dense layer that will accept the extracted features and will 
    then pass them to the FUSING Network during the Merge operation.
    
    Output: 2048-d vector
    """
    @staticmethod
    def thumbnail_submodel(model_structure_type='functional'):
        # declare input shape
        input_shape = (2048,)

        if model_structure_type == 'functional':
            # Declare Thumbnail Input
            thumbnail_in = Input(shape=input_shape, name='thumbnail_input')
            thumbnail_out = Dense(32, activation="relu", name="thumbnail")(thumbnail_in)
            return thumbnail_out
        elif model_structure_type == 'sequential':
            # Create Model
            model = Sequential()

            # Declare Thumbnail Input
            model.add(Input(shape=input_shape, name='thumbnail_input'))
            model.add(Dense(32, activation="relu", name="thumbnail"))
            # model.add(Dropout(0.5))
            return model

    """
    Model that accepts as input the thumbnail contents vector and fine-tunes an Inception V3 CNN. 

    Output: 2048-d vector
    """
    @staticmethod
    def thumbnail_CNN_submodel():
        # Add the CNN Inception V3 model
        cnn = InceptionV3(weights='imagenet', include_top=False)

        # get layers and add average pooling layer
        cnn_out = cnn.output

        # Add an Average Pooling Layer
        average_pool_out = GlobalAveragePooling2D(name='average_pooling_layer')(cnn_out)
        dense_output = Dense(32, activation="relu", name="thumbnail")(average_pool_out)

        model = Model(inputs=cnn.input, outputs=dense_output)

        # Freeze pre-trained CNN model area's layer
        for layer in cnn.layers:
            layer.trainable = False

        return model

    """
    Model that accepts as input the HEADLINES FEATURES of size (SET_LENGTH x MAX_WORDS_IN HEADLINE) and  
    processes the HEADLINE in order the extract useful representations from the words in it. 
    It includes and Embedding layer and an LSTM that accepts as a sequence each word of the headline.
    
    Output: 100-d vector
    """
    def headline_submodel(self):
        # Init Embedding Layer variables
        _vocab_size = self._headlines_vocab_size
        # _vocab_size = 1000 # equivalent to the max_features parameter of the CountVectorizer in data_preparation
        embedding_vector_length = 32

        # Init LSTM variables
        lstm_headline_units = embedding_vector_length
        lstm_headline_input_shape = (self._headlines_words_seq_length, embedding_vector_length)

        # Create Model
        model = Sequential()
        # Add Embedding Layer
        model.add(Embedding(_vocab_size + 1,
                            embedding_vector_length,
                            input_length=self._headlines_words_seq_length,
                            trainable=True,
                            name='headline_embedding_layer')) # TRUE: update weights during training

        # Add LSTM
        model.add(LSTM(lstm_headline_units,
                       activation='tanh',
                       kernel_initializer='glorot_uniform', # 'uniform' or 'glorot_uniform'
                       go_backwards=False,
                       return_sequences=False,
                       input_shape=lstm_headline_input_shape,
                       dropout=self.text_input_dropout_level,
                       recurrent_dropout=0.0,
                       name='headline_lstm'))
        # model.add(Dropout(0.5))
        return model

    """
    Model that accepts as input the preprocessed statistics of each video of size (1x5) and uses a dense layer to 
    extract a feature representation from them that will be then passed to the FUSING Network.
    
    Output 1: 5-d vector for statistics features only
    Output 2: 25-d vector for all style features    
    """
    def statistics_submodel(self):
        # Init variables
        if self._other_features_type == 'statistics':
            dense_statistics_units = 5
            dense_dtype = 'float32'
        elif self._other_features_type == 'all':
            dense_statistics_units = 25
            dense_dtype = 'float32'

        statistics_dense_input_shape = (dense_statistics_units,)

        # Create Model
        model = Sequential()
        #Add a Dense Layer
        
        model.add(Dense(32,
                        input_shape=statistics_dense_input_shape,
                        activation='relu',
                        kernel_initializer='glorot_uniform', # 'uniform' or 'glorot_uniform'
                        dtype=dense_dtype,
                        name='statistics_fully_connected'))
        # model.add(Dropout(0.5))
        return model

    """
    Model that accepts as input the VIDEO TAGS FEATURES of size (SET_LENGTH x MAX_TAGS_IN_A_VIDEO)  
    and processes the TAGS in order to extract useful representations between them in a video. 
    It includes and Embedding layer and an LSTM that accepts as a sequence each video tag.

    Output: 100-d vector
    """
    def video_tags_submodel(self):
        # Init Embedding Layer variables
        _vocab_size = self._video_tags_vocab_size
        embedding_vector_length = 32

        # Init LSTM variables
        lstm_video_tags_units = embedding_vector_length
        lstm_video_tags_input_shape = (self._video_tags_seq_length, embedding_vector_length)

        # Create Model
        model = Sequential()

        # Add Embedding Layer
        model.add(Embedding(_vocab_size + 1,
                            embedding_vector_length,
                            input_length=self._video_tags_seq_length,
                            trainable=True,
                            name='video_tags_embedding_layer'))

        # Add LSTM
        model.add(LSTM(lstm_video_tags_units,
                       activation='tanh',
                       kernel_initializer='glorot_uniform',
                       # go_backwards=False,
                       return_sequences=False,
                       input_shape=lstm_video_tags_input_shape,
                       dropout=self.text_input_dropout_level,
                       recurrent_dropout=0.0,
                       name='video_tags_lstm'))
        return model
    def  custom_standardization(self, input_data):
        lowercase = tf.strings.lower(input_data)
        stripped_html = tf.strings.regex_replace(lowercase, "\'", "'")
        return tf.strings.regex_replace(
            stripped_html, "[%s]" % re.escape(string.punctuation), ""
        )
    
    def subtitle_submodel(self):

        max_features = 20000
        embedding_dim = 128
        sequence_length = 5000
        vectorize_layer = TextVectorization(
                        standardize=self.custom_standardization,
                        max_tokens=max_features,
                        output_mode="int",
                        output_sequence_length=sequence_length,
                        )
        vectorize_layer.adapt(self._train_tokens)
        # 'embedding_dim'.
        model = tf.keras.models.Sequential()
        model.add(tf.keras.Input(shape=(1,), dtype=tf.string))
        model.add(vectorize_layer)
        model.add(Embedding(max_features, embedding_dim))
        model.add(Conv1D(28, 9, padding="same", activation="relu", strides=3))
        model.add(Conv1D(28, 9, padding="same", activation="relu", strides=3))
        model.add(Conv1D(28, 9, padding="same", activation="relu", strides=3))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(32, activation="relu", name="subtitle_branch"))
        # model.add(Dropout(0.5))
        return model

    """
    Method that takes ALL different sub-models and builds the whole model. It also splits the input to the 
    different parts required to be fed to each submodel. At the end it concats all the outputs of the sub-models
    and pass that to a Fully connected layer and then to a Softmax layer that performs the classification.
    """
    def build_all_inputs_full_model(self):
        print('\n-------------------------------------------------')
        print('---      Started building the FULL MODEL      ---')
        print('-------------------------------------------------\n')

        """
        Get the THUMBNAIL branch model
        """
        thumbnail_branch = self.thumbnail_submodel(model_structure_type='sequential')
        # thumbnail_branch = self.thumbnail_CNN_submodel()

        """
        Get the HEADING (TITLE) branch model
        """
        heading_branch = self.headline_submodel()

        """
        Get the STATISTICS branch model
        """
        statistics_branch = self.statistics_submodel()

        """
        Get the VIDEO TAGS branch model
        """
        video_tags_branch = self.video_tags_submodel()

        """
        Subtitle submodel 
        """
        subtitle_branch = self.subtitle_submodel()
        """
        Merge together all Branches
        """
        # merged_inputs = Concatenate(name='x_merged_inputs')([thumbnail_branch.output, heading_branch.output, statistics_branch.output, video_tags_branch.output, subtitle_branch.output])
        merged_inputs = Average(name='x_merged_inputs')([thumbnail_branch.output, heading_branch.output, statistics_branch.output, video_tags_branch.output, subtitle_branch.output])
        """
        Add multiple Fully-Connected DENSE Layers for Dimensionality Reduction
        """
        # fully_connected_3 = Dense(512, activation='relu')(merged_inputs)
        fully_connected_3 = merged_inputs
        """
        Add a DROPOUT layer before the classification layer
        """
        if self.dimensionality_reduction_layers:
            dropout_layer = Dropout(self.final_dropout_level)(fully_connected_3)
        else:
            # dropout_layer = Dropout(self.final_dropout_level)(fully_connected_0)
            dropout_layer = Dropout(self.final_dropout_level)(merged_inputs)

        # Add a SOFTMAX layer at the end to perform the classification

        
        if self.loss_function == 'binary_crossentropy':
            all_inputs_out = Dense(1, activation="sigmoid", name="classification_layer")(dropout_layer)
        else:
            all_inputs_out = Dense(self.nb_classes,
                               activation='softmax',
                               name='classification_layer')(dropout_layer)

        """
        Create the Functional Model and declare Input and Output
        """
        model_inputs = list()
        model_inputs.append(thumbnail_branch.input)
        model_inputs.append(heading_branch.input)
        model_inputs.append(statistics_branch.input)
        model_inputs.append(video_tags_branch.input)
        model_inputs.append(subtitle_branch.input)
        full_model = Model(model_inputs, all_inputs_out)

        """
        Choose an Optimizer
        """
        # Declare an Optimizer
        optimizer = Adam(learning_rate=self.learning_rate,
                         beta_1=self.adam_beta_1,
                         beta_2=self.adam_beta_2,
                         decay=self.decay,  # learning rate decay over each update.
                         epsilon=self.epsilon,  # Fuzz factor. If None, defaults to K.epsilon()
                         amsgrad=False)  # whether to apply AMSGrad variant of Adam algorithm or not

        """
        Compile the Model
        """
        full_model.compile(loss=self.loss_function, optimizer=optimizer, metrics=self.metrics)

        """
        Summarize the Model
        """
        print(full_model.summary())
        return full_model
