import os
import os.path
import pandas as pd
import numpy as np
from skimage import io
import pickle
import datetime
import emoji
from skimage.transform import resize
import isodate as isodate
# cv2
import threading
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import InceptionV3

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import one_hot, text_to_word_sequence

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.utils import to_categorical

from tensorflow.keras import *
from tensorflow.keras.models import *
from tensorflow.keras.preprocessing import image
from tensorflow.keras import backend as K
from tensorflow.keras import metrics
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.utils import Sequence

from random import randint
import random

class CNN_MODEL(object):
    def __init__(self):
        """
        C'tor
        """

        # Set image input shape
        self.image_input_shape_channels = (299, 299, 3)
        self.image_input_shape = (299, 299)

        # Get CNN model with pre-trained ImageNet weights. Include also the top fully connected layer
        # and we will freeze the last softmax classification layer in the next line
        CNN_model = InceptionV3(weights='imagenet',
                                include_top=True)

        # Freeze the last softmax layer. We'll extract features at the final pool layer.
        self.model = Model(inputs=CNN_model.input,
                           outputs=CNN_model.get_layer('avg_pool').output)

    def extract_features_image(self, frame_image_path, isEmptyImage=False):
        """
        Method that takes as input the path of an image (in our case a video THUMBNAIL)
        converts it to a numpy array using Keras.preprocessing package and then extracts the features from that
        image frame by running it into the pre-trained CNN Inception V3 model
        :param frame_image_path: the path to get the image file
        :param isEmptyImage: True if there is no thumbnail file, otherwise False
        :return: the thumbnail image features
        """
        if not isEmptyImage:
            # Read image
            img = image.load_img(frame_image_path, target_size=self.image_input_shape) # only requires wxh size and adds channels by default
            x = image.img_to_array(img) # 299x299x3
            x = np.expand_dims(x, axis=0) # 1x299x299x3
            x = preprocess_input(x) # 1x299x299x3
        else:
            # Create an image zero numpy array of size (299x299x3)
            img = np.zeros(self.image_input_shape_channels)
            x = np.expand_dims(img, axis=0)
            x = preprocess_input(x)

        # Get the prediction.
        features = self.model.predict(x)
        features = features[0]

        return features
    
def get_num_of_emoticons(str):
        """
        Method that calculates the number of emoticons in the given text
        :param str: the string to be checked
        :return: the number of emoticons in the given string
        """
        num_emoticons = 0

        for character in str:
            if character in emoji.UNICODE_EMOJI:
                num_emoticons += 1

        return num_emoticons
    
def get_jaccard_similarity(str1, str2):
        """
        Method that calculates the Jaccard Similarity between two strings
        :param str1: the first string
        :param str2: the second string
        :return: the jaccard simillarity of the two provided strings
        """
        a = set(str1.split())
        b = set(str2.split())
        c = a.intersection(b)
        jaccard_sim = float(len(c)) / (len(a) + len(b) - len(c))

        return float("{0:.2f}".format(jaccard_sim))
def preprocess_headlines_one_hot( video_headline, vocab_size=10277, headline_max_words_length=21):
        """
        Method that extracts the features of a given video headline
        :param video_headline: the video's Headline
        :param vocab_size:
        :param headline_max_words_length: the bigger headline size from all the headlines used to train the classifier
        :return: the preprocessed one-hot encoded headline
        """

        # Integer encode the document
        encoded_headline = one_hot(video_headline, vocab_size)

        # Perform padding
        headline_features = sequence.pad_sequences(np.expand_dims(encoded_headline, axis=0), maxlen=headline_max_words_length)

        return np.array(headline_features)
def preprocess_video_tags_one_hot( video_tags, vocab_size=19040, video_tags_max_length=95):
        """
        Method that performs the required pre-processing for the video tags of a given YouTube Video ID
        using keras.preprocessing.text.one_hot. It is actually a bag-of-words technique.
        :param video_tags: the list with all the video tags to be processed
        :param vocab_size: the vocabulary size of all the videos' video tags used to train the classifier
        :param video_tags_max_length: the bigger video tags size from all the headlines used to train the classifier
        :return: the preprocessed one-hot encoded video_tags numpy array
        """
        # init variables
        final_encoded_video_tags, encoded_video_tags = list(), list()

        if (not isinstance(video_tags, list)) or (len(video_tags) == 0):
            # this video has not tags so add an empty array
            final_encoded_video_tags.append([])
        else:
            # encode each tag of the video separately
            for tag in video_tags:
                encoded_video_tags += one_hot(tag, vocab_size)

            # append the current video's encoded video tags array
            final_encoded_video_tags.append(encoded_video_tags)

        # Perform padding
        video_tags_features = sequence.pad_sequences(final_encoded_video_tags, maxlen=video_tags_max_length)

        return np.array(video_tags_features)
    
def get_video_general_style_features(video_information):
        """
        Method that extracts the style (other- and statistics-related) features for a given YouTube Video's Information
        :param video_information: the dictionary with all the video information to be used
        :return: a numpy-like array with all the video's style features
        """
        # init variables
        video_general_style_features = list()
        bad_words = ['sex', 'undress', 'kiss', 'kill', 'killed', 'smoke', 'weed', 'burn', 'die', 'dead', 'death', 'burried',
                     'alive', 'suicide', 'poop', 'inject', 'injection', 'arrested', 'hurt', 'naked', 'blood', 'bloody']

        kids_related_words = ['tiaras', 'kid', 'kids', 'toddler', 'toddlers', 'surprise', 'fun', 'funny', 'disney',
                              'school', 'learn', 'superheroes', 'heroes', 'family', 'baby', 'mickey']

        """
        Get Video-related Features
        """
        # Get Video's duration in seconds
        try:
            # duration = video_information.video_duration
            # video_duration = isodate.parse_duration(duration)
            # video_duration_in_seconds = video_duration.total_seconds()
            video_duration_in_seconds = video_information.video_duration
        except AttributeError:
            video_duration_in_seconds = 0
        # Add it to the result
        video_general_style_features.append(int(video_duration_in_seconds))

        # Get Video's Category
        try:
            video_categoryId = int(video_information.category_id)
        except AttributeError:
            video_categoryId = 0
        video_general_style_features.append(video_categoryId)

        """
        Get Video Statistics-related Features
        """
        # get video views
        try:
            video_views_cntr = 0
            if str(video_information.view_count).isdigit():
                video_views_cntr = int(video_information.view_count)
        except AttributeError:
            video_views_cntr = 0
        # get video likes
        try:
            video_likes_cntr = 0
            if str(video_information.like_count).isdigit():
                video_likes_cntr = int(video_information.like_count)
        except AttributeError:
            video_likes_cntr = 0
        # get video dislikes
        try:
            video_dislikes_cntr = 0
            if str(video_information.dislike_count).isdigit():
                video_dislikes_cntr = int(video_information.dislike_count)
        except AttributeError:
            video_dislikes_cntr = 0
        if video_dislikes_cntr > 0:
            video_likes_dislikes_ratio = int(video_likes_cntr / video_dislikes_cntr)
        else:
            video_likes_dislikes_ratio = video_likes_cntr
        try:
            video_comments_cntr = 0
            if str(video_information.comment_count).isdigit():
                video_comments_cntr = int(video_information.comment_count)
        except AttributeError:
            video_comments_cntr = 0

        # Add all of them to the result
        video_general_style_features.append(video_views_cntr)
        video_general_style_features.append(video_likes_cntr)
        video_general_style_features.append(video_dislikes_cntr)
        video_general_style_features.append(video_likes_dislikes_ratio)
        # video_general_style_features.append(video_added_favourites_cntr)
        video_general_style_features.append(video_comments_cntr)


        """
        Get Video Title- and description-related Features
        """
        # get video title and split it into words
        video_title = video_information.title
        words_in_video_title = video_title.split()
        # get video description and split it into words
        video_description = str(video_information.description)
        words_in_video_description = video_description.split()

        # get title length
        video_title_length = len(words_in_video_title)
        # get description length
        video_description_length = len(words_in_video_description)
        # get description ratio over the title
        video_description_title_ratio = int(video_description_length / video_title_length)
        # get jaccard similarity between words appearing in title and description
        video_description_title_jaccard_similarity = get_jaccard_similarity(video_description, video_title)

        # get number of exclamation and question marks in title
        video_title_exclamation_marks_cntr = video_title.count('!')
        video_title_question_marks_cntr = video_title.count('?')
        # get number of emoticons in title
        video_title_emoticons_cntr = get_num_of_emoticons(video_title)
        # get number of bad words in title
        video_title_bad_words_cntr = 0
        for word in words_in_video_title:
            if word.lower() in bad_words:
                video_title_bad_words_cntr += 1
        # get number of kids-related words in title
        video_title_kids_related_words_cntr = 0
        for word in words_in_video_title:
            if word.lower() in kids_related_words:
                video_title_kids_related_words_cntr += 1

        # get number of exclamation and question marks in description
        video_description_exclamation_marks_cntr = video_description.count('!')
        video_description_question_marks_cntr = video_description.count('?')
        # get number of emoticons in description
        video_description_emoticons_cntr = get_num_of_emoticons(video_description)
        # get number of bad words in description
        video_description_bad_words_cntr = 0
        for word in words_in_video_description:
            if word.lower() in bad_words:
                video_description_bad_words_cntr += 1
        # get number of kids-related words in title
        video_description_kids_related_words_cntr = 0
        for word in words_in_video_description:
            if word.lower() in kids_related_words:
                video_description_kids_related_words_cntr += 1

        # Add all of them to the result
        video_general_style_features.append(video_title_length)
        video_general_style_features.append(video_description_length)
        video_general_style_features.append(video_description_title_ratio)
        video_general_style_features.append(video_description_title_jaccard_similarity)

        video_general_style_features.append(video_title_exclamation_marks_cntr)
        video_general_style_features.append(video_title_question_marks_cntr)
        video_general_style_features.append(video_title_emoticons_cntr)
        video_general_style_features.append(video_title_bad_words_cntr)
        video_general_style_features.append(video_title_kids_related_words_cntr)

        video_general_style_features.append(video_description_exclamation_marks_cntr)
        video_general_style_features.append(video_description_question_marks_cntr)
        video_general_style_features.append(video_description_emoticons_cntr)
        video_general_style_features.append(video_description_bad_words_cntr)
        video_general_style_features.append(video_description_kids_related_words_cntr)


        """
        Get Video Tags-related Features
        """
        try:
            video_tags = video_information.tags
        except AttributeError:
            video_tags = []

        # get number of tags in the video
        video_tags_cntr = len(video_tags)
        # init other features variables in case this video has no tags
        video_tags_bad_words_cntr = 0
        video_tags_kids_related_words_cntr = 0
        video_tags_title_jaccard_similarity = 0

        if video_tags_cntr > 0:
            # get number of bad words in video tags
            for tag in video_tags:
                # check if there is more than 1 word in the current video tag
                if ' ' in tag:
                    words_in_current_tag = tag.split()
                    for word in words_in_current_tag:
                        if word.lower() in bad_words:
                            video_tags_bad_words_cntr += 1
                else:
                    if tag.lower() in bad_words:
                        video_tags_bad_words_cntr += 1

            # get number of kids-related words in video tags
            for tag in video_tags:
                # check if there is more than 1 word in the current video tag
                if ' ' in tag:
                    words_in_current_tag = tag.split()
                    for word in words_in_current_tag:
                        if word.lower() in bad_words:
                            video_tags_kids_related_words_cntr += 1
                else:
                    if tag.lower() in bad_words:
                        video_tags_kids_related_words_cntr += 1

            # get Jaccard similarity between words appearing in video tags and description
            # create a text string from all the tags
            video_tags_text = ''
            for tag in video_tags:
                video_tags_text += tag + ' '
            video_tags_title_jaccard_similarity = get_jaccard_similarity(video_tags_text, video_title)

        # Add everything to the result
        video_general_style_features.append(video_tags_cntr)
        video_general_style_features.append(video_tags_bad_words_cntr)
        video_general_style_features.append(video_tags_kids_related_words_cntr)
        video_general_style_features.append(video_tags_title_jaccard_similarity)

        """
        Convert the result array to a numpy array
        """
        final_video_general_style_features = np.expand_dims(np.asarray(video_general_style_features), axis=0)
        final_video_general_style_features = Normalization(axis=-1)(final_video_general_style_features.astype("float64"))

        return final_video_general_style_features
    
def PRECISION(y_true, y_pred):
    """
    Precision metric.
    Only computes a batch-wise average of precision.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def RECALL(y_true, y_pred):
    """
    Recall metric.
    Only computes a batch-wise average of recall.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def F1_SCORE(y_true, y_pred):
    """
    F1 Score metric.
    Only computes a batch-wise average of recall.
    """
    p = PRECISION(y_true, y_pred)
    r = RECALL(y_true, y_pred)
    return (2 * p * r) / (p + r + K.epsilon())

def AUC_SCORE(y_true, y_pred):
    """
    ROC_AUC Score metric.
    Only computes a batch-wise average of roc_auc.
    """
    auc = tf.metrics.auc(y_true, y_pred, curve='ROC')[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

def shuffle_weights(model):
    weights = [glorot_uniform(seed=41)(w.shape) if w.ndim > 1 else w for w in model.get_weights()]
    model.set_weights(weights)
 
