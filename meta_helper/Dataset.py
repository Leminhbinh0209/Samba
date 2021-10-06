# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 20:33:33 2018

@author: kostantinos.papadamou

This class implements all the functions that are required to retrieve all the data from our Dataset (MongoDB)
and those stored to the disk and makes all the pre-processing required before training our model.
"""

import random

import isodate as isodate
from PIL import Image
from tqdm import tqdm
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.utils import shuffle
import numpy
import os
import pickle
import glob
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import one_hot, text_to_word_sequence
from tensorflow.keras.preprocessing import sequence
import threading
import unicodedata
import emoji
import re
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input


class DATASET(object):
    #
    # CONSTRUCTOR
    #
    def __init__(self, prepare_dataset_crossvalidation=False, nb_classes=2):
        """
        Set the Base Directories
        """
        self.DATASET_DATA_BASE_DIR = './data/'

        self.THUMBNAILS_DIR = 'thumbnails/'
        self.THUMBNAILS_FEATURES_DIR = self.DATASET_DATA_BASE_DIR + 'thumbnails_features/'
        self.STATISTICS_FEATURES_BASE_DIR = self.DATASET_DATA_BASE_DIR + 'statistics_features/'
        self.HEADLINES_FEATURES_BASE_DIR = self.DATASET_DATA_BASE_DIR + 'headlines_features/'
        self.VIDEO_TAGS_FEATURES_BASE_DIR = self.DATASET_DATA_BASE_DIR + 'video_tags_features/'

        """
        MongoDB Configuration
        """
        # Host and Port
        self.client = MongoClient('localhost', 27017)

        # DB Name
        self.db = self.client['y4kids']

        # COLLECTIONS Names
        self.videos_col = self.db.groundtruth_videos
        self.dataset_metadata_col = self.db.dataset_metadata

        # Set the classes
        self.nb_classes = nb_classes
        if nb_classes == 2:
            self.classes = ['suitable', 'restricted']
        else:
            self.classes = ['suitable', 'disturbing', 'restricted', 'irrelevant']

        # Get Data (Video IDs) for TRAIN and TEST sets
        if prepare_dataset_crossvalidation:
            self.videos_set = self.get_dataset_video_ids(corpus='all_crossvalidation', ignore_videos_with_disagreements=True)
            self.videos_set_size = len(self.videos_set)
        return

    """
    Method that retrieves all Video IDs for videos that are in our Dataset corpus
    corpus can take one of the following values:
    1. all
    2. train
    3. test
    """
    def get_dataset_video_ids(self,
                              corpus="all",
                              ignore_videos_with_disagreements=False,
                              num_examples=62600):
        print('--- Retrieving all Dataset Video IDs for corpus=' + corpus + '...')

        # Construct the Query to retrieve only the videos that have been selected during the sampling process
        # Get only the video id for each video
        retrieved_videos = None
        if corpus == 'all_crossvalidation':
            retrieved_videos = self.dataset_metadata_col.find_one(
                {'metadata_type': 'train_test_sets'},
                {
                    'videos_set': 1,
                    'labels_set': 1
                }
            )
        elif corpus == 'whole_dataset':
            retrieved_videos = self.videos_col.find({}, {'id': 1, 'selectedFromSamplingProcess': 1})

        # Get all retrieved video ids
        dataset_video_ids = []

        # For TRAINing and TESTing ignore Video IDs that has a NULL label
        if corpus == 'all_crossvalidation':
            if ignore_videos_with_disagreements:
                # IGNORE videos where we have disagreements and they have a NULL label
                iter_cntr = 0
                for video in retrieved_videos['videos_set']:
                    if retrieved_videos['labels_set'][iter_cntr] is not None:
                        dataset_video_ids.append(video['video_id'])
                    iter_cntr += 1
            else:
                # Add all videos the set
                for video in retrieved_videos['videos_set']:
                    dataset_video_ids.append(video['video_id'])
        elif corpus == 'whole_dataset':
            dataset_size = retrieved_videos.count()
            num_videos_to_exclude = dataset_size - num_examples

            # Add videos to the the final video ids list excluding the required number of unlabeled data given
            # the requested num_examples
            for video in retrieved_videos:
                # First check if the current videos is in our Ground Truth
                # If it is not then don't add videos to the final list if the num_videos_to_exclude is > 0
                if self.key_exists(video, 'selectedFromSamplingProcess'):
                    dataset_video_ids.append(video['id'])
                else:
                    # video is not in our Ground Truth so check if we should add it or not
                    if num_videos_to_exclude == 0:
                        # Add this video to the final Video IDs list
                        dataset_video_ids.append(video['id'])
                    else:
                        # reduce the number of the videos that we need to exclude since we are excluding the current video
                        num_videos_to_exclude = num_videos_to_exclude - 1

        return dataset_video_ids

    """
    Method that retrieves from mongodb the various other data that will be inputted
    to our ML model. For example we add different statistics like view count and 
    counters regarding the comments i.e., comments that contain the word bait
    """
    def preprocess_video_statistics(self, set_video_ids):
        # Declare variables
        results = []

        # Declare a progress bar
        progressbar = tqdm(total=len(set_video_ids))

        # For each video in the provided set (TRAIN or TEST) retrieve all statistics
        for video_id in set_video_ids:
            # Get video details from MongoDB for the current Video ID
            video = self.videos_col.find_one({'id': video_id}, {'id': 1, 'statistics': 1})

            # get views counter
            try:
                views_count = video['statistics']['viewCount']
            except KeyError:
                views_count = 0
                pass
            # get likes counter
            try:
                likes_count = video['statistics']['likeCount']
            except KeyError:
                likes_count = 0
                pass
            # get comments counter
            try:
                comments_count = video['statistics']['commentCount']
            except KeyError:
                comments_count = 0
                pass
            # get favorite counter
            try:
                favorites_count = video['statistics']['favoriteCount']
            except KeyError:
                favorites_count = 0
                pass
            # get dislikes counter
            try:
                dislikes_count = video['statistics']['dislikeCount']
            except KeyError:
                dislikes_count = 0
                pass

            # Add current video's retrieved information in an Array
            video_stats = list()
            video_stats.append(views_count)
            video_stats.append(likes_count)
            video_stats.append(comments_count)
            video_stats.append(favorites_count)
            video_stats.append(dislikes_count)

            # Add current video's retrieved stats to the result array
            results.append(video_stats)

            # Update progress bar
            progressbar.update(1)

        # Finish progress bar
        progressbar.close()

        return numpy.asarray(results)


    """
    Method that retrieves the preprocessed video statistics from the disk for 
    the given set (TRAIN or TEST)
    """
    def get_video_statistics_features_for_training(self, set_type, is_crossvalidation=True):
        # Create the absolute path to the video statistics features file
        if is_crossvalidation:
            filename = self.STATISTICS_FEATURES_BASE_DIR + 'all_statistics_features.p'
        else:
            filename = self.STATISTICS_FEATURES_BASE_DIR + set_type + '/video_statistics_' + set_type + '_set.p'

        # Retrieve the video statistics from the disk as a numpy array
        preprocessed_video_statistics = pickle.load(open(filename, 'rb'))
        return preprocessed_video_statistics

    """
    Method that returns an Array with all the Labels of the given Video IDs
    """
    def get_labels(self,
                   video_ids,
                   include_video_ids=False,
                   include_age_restricted=False,
                   include_annotated_age_restricted=False,
                   label_binary_encoding=False,
                   store_to_db=False):
        print("--- Started getting Video LABELS...")

        # Declare variables
        dataset_video_labels = []
        isVideoAgeRestricted = False

        progressbar = tqdm(total=len(video_ids))

        """
        For each Video ID retrieve its LABEL
        """
        for video_id in video_ids:
            # Get all annotations for the current Video ID
            video_info = self.videos_col.find_one({'id': video_id}, {'classification': 1, 'contentDetails': 1})

            """
            First check if it is Age Restricted
            """
            if self.key_exists(video_info, 'contentDetails', 'contentRating', 'ytRating') and not include_annotated_age_restricted:
                # Check if we want to include Age Restricted videos
                if not include_age_restricted:
                    # Update progress bar state
                    progressbar.update(1)
                    continue # SKIP this video since we don't want Age Restricted Videos

                if video_info['contentDetails']['contentRating']['ytRating'] == 'ytAgeRestricted':
                    if self.nb_classes == 2: # 2-class classification
                        # Set it as DISTURBING
                        if label_binary_encoding:
                            video_label = 1
                        else:
                            video_label = 'disturbing'
                    elif self.nb_classes == 4: # 4-class classification
                        # Set it as COMPLETELY INAPPROPRIATE
                        if label_binary_encoding:
                            video_label = 2
                        else:
                            video_label = 'inappropriate'  # inappropriate = 2

                    # set the flag to let the loop know that the current video is Age Restricted
                    isVideoAgeRestricted = True
                else:
                    isVideoAgeRestricted = False

            """
            If the current video is not Age Restricted get its label in the normal way
            """
            if not isVideoAgeRestricted:
                # It is not Age Restricted so check the annotations
                # Parse all annotations of the current Video ID and find the correct LABEL for that Video
                appropriate_cntr = 0
                disturbing_cntr = 0
                inappropriate_cntr = 0
                other_cntr = 0

                # Check if the video has any annotations otherwise set its label as 4.0 (unlabeled)
                if self.key_exists(video_info, 'classification'):
                    for annotation_info in video_info['classification']:
                        if annotation_info['category'] == 'appropriate':
                            appropriate_cntr += 1
                        elif annotation_info['category'] == 'disturbing':
                            disturbing_cntr += 1
                        elif annotation_info['category'] == 'inappropriate':
                            inappropriate_cntr += 1
                        elif annotation_info['category'] == 'other-irrelevant':
                            other_cntr += 1

                    # Accept as LABEL the category with the more annotations
                    maximum_annotations = max(appropriate_cntr, disturbing_cntr, inappropriate_cntr, other_cntr)
                    if maximum_annotations > 1:
                        # Set the video LABEL to the class with the max annotations
                        if appropriate_cntr == maximum_annotations: # APPROPRIATE
                            if label_binary_encoding:
                                video_label = 0
                            else:
                                video_label = 'appropriate'

                        elif disturbing_cntr == maximum_annotations: # DISTURBING
                            if label_binary_encoding:
                                video_label = 1
                            else:
                                video_label = 'disturbing'

                        elif inappropriate_cntr == maximum_annotations: # COMPLETELY INAPPROPRIATE
                            if self.nb_classes == 2: # 2-class classification
                                if label_binary_encoding:
                                    video_label = 1
                                else:
                                    video_label = 'disturbing'
                            elif self.nb_classes == 4:  # 4-class classification
                                if label_binary_encoding:
                                    video_label = 2
                                else:
                                    video_label = 'inappropriate'

                        elif other_cntr == maximum_annotations: # OTHER_IRRELEVANT
                            if self.nb_classes == 2: # 2-class classification
                                if label_binary_encoding:
                                    video_label = 0
                                else:
                                    video_label = 'appropriate'
                            elif self.nb_classes == 4: # 4-class classification
                                if label_binary_encoding:
                                    video_label = 3
                                else:
                                    video_label = 'other-irrelevant'

                        else:
                            # Do that for all N-class classification
                            if self.nb_classes == 2: # 2-class classification
                                if label_binary_encoding:
                                    video_label = 2
                                else:
                                    video_label = None # Unlabeled = 2
                            elif self.nb_classes == 4: # 4-class classification
                                if label_binary_encoding:
                                    video_label = 4
                                else:
                                    video_label = None # Unlabeled = 4
                    else:
                        # Do that for all N-class classification
                        if self.nb_classes == 2: # 2-class classification
                            if label_binary_encoding:
                                video_label = 2
                            else:
                                video_label = None  # Unlabeled = 2
                        elif self.nb_classes == 4: # 4-class classification
                            if label_binary_encoding:
                                video_label = 4
                            else:
                                video_label = None  # Unlabeled = 4
                        # print('ERROR: There is no agreement on the labels. Skip that video [SKIP]')
                else:
                    # Do that for all N-class classification
                    if self.nb_classes == 2: # 2-class classification
                        if label_binary_encoding:
                            video_label = 2
                        else:
                            video_label = None  # Unlabeled = 2
                    elif self.nb_classes == 4: # 4-class classification
                        if label_binary_encoding:
                            video_label = 4
                        else:
                            video_label = None  # Unlabeled = 4

            # Store label to MongoDB
            if store_to_db:
                self.videos_col.update_one({'id': video_id}, {"$set": {'classification_label': video_label}}, upsert=False)

            # Add the label to the result array
            if include_video_ids:
                video = {}
                video['video_id'] = video_id
                video['label'] = video_label
                dataset_video_labels.append(video)
            else:
                dataset_video_labels.append(video_label)

            # Reset Age Restricted flag
            isVideoAgeRestricted = False

            # Update progress bar state
            progressbar.update(1)

        progressbar.close()
        return dataset_video_labels

    """
    Method that retrieves the labels for the whole Dataset and performs one-hot categorical encoding 
    for each one and in the end it returns a numpy array which includes all the encoded labels for the given set
    """
    def get_labels_one_hot_encoded_for_crossvalidation_training(self, ignore_video_with_disagreements=False, perform_one_hot=True):
        print('--- Getting DATASET labels for training...')

        # Get the appropriate set's Video IDs
        # if self.nb_classes == 2 or self.nb_classes == 4:
        set_labels = self.dataset_metadata_col.find_one(
            {
                'metadata_type': 'train_test_sets',
                'nb_classes': self.nb_classes
            },
            {'labels_set': 1}
        )
        all_set_labels = set_labels['labels_set']

        # Init variables
        y = []
        video_counter = 0

        # Perform one-hot categorical encoding on each label and append it to the result
        for label in all_set_labels:
            if ignore_video_with_disagreements:
                if label is not None:
                    # Get the one-hot categorical encoding of current video's label
                    if perform_one_hot:
                        y.append(self.get_class_one_hot(label))
                    else:
                        y.append(self.get_class_to_categorical(label))
            else:
                # Get the one-hot categorical encoding of current video's label
                if perform_one_hot:
                    y.append(self.get_class_one_hot(label))
                else:
                    y.append(self.get_class_to_categorical(label))

            # increase video counter
            video_counter += 1

        return numpy.array(y)

    """
    Method that returns two arrays one with all the Video IDs for each one video in our Ground Truth Dataset and 
    the second array includes the corresponding Labels for each video. If we requested it also stores this information
    in the database
    """
    def prepare_dataset_for_crossvalidation(self,
                                            include_age_restricted_videos=False,
                                            ignore_videos_with_disagreement=True,
                                            include_annotated_age_restricted=False,
                                            store_to_db=False,
                                            include_video_ids_in_labels=False):
        """
        Get all Video IDs in our Ground Truth
        """
        all_video_ids = self.get_dataset_video_ids(corpus='all')

        """
        Get the labels for all the videos in our Ground Truth
        """
        all_video_labels = self.get_labels(all_video_ids,
                                           include_video_ids=False,
                                           include_age_restricted=include_age_restricted_videos,
                                           include_annotated_age_restricted=include_annotated_age_restricted,
                                           label_binary_encoding=False,
                                           store_to_db=False)

        """
        Get the Indices of Videos of each Class
        """
        if self.nb_classes == 2:
            # Get the indices of the Appropriate videos
            indices_appropriate = [i for i, x in enumerate(all_video_labels) if x == 'appropriate' or x == 'other-irrelevant']
            # Get the indices of the Disturbing videos
            indices_disturbing = [i for i, x in enumerate(all_video_labels) if x == 'disturbing' or x == 'inappropriate']

            # Concate all indices together so that we have an array with all the videos in our
            # Ground Truth Dataset but this time sorted by class
            all_videos_indices_sorted_by_class = indices_appropriate + indices_disturbing
        elif self.nb_classes == 4:
            # Get the indices of the Appropriate videos
            indices_appropriate = [i for i, x in enumerate(all_video_labels) if x == 'appropriate']
            # Get the indices of the Disturbing videos
            indices_disturbing = [i for i, x in enumerate(all_video_labels) if x == 'disturbing']
            # Get the indices of the Inappropriate videos
            indices_inappropriate = [i for i, x in enumerate(all_video_labels) if x == 'inappropriate']
            # Get the indices of the Other videos
            indices_other = [i for i, x in enumerate(all_video_labels) if x == 'other-irrelevant']

            # Concate all indices together so that we have an array with all the videos in our
            # Ground Truth Dataset but this time sorted by class
            all_videos_indices_sorted_by_class = indices_appropriate + indices_disturbing + indices_inappropriate + indices_other

        # Add Videos with disagreements if requested
        if not ignore_videos_with_disagreement:
            # Get the indices of the Unknown videos
            indices_unknown = [i for i, x in enumerate(all_video_labels) if x is None]
            all_videos_indices_sorted_by_class += indices_unknown

        """
        Now that we have all the indices of all videos sorted by class
        we can proceed and create the actual arrays with all the Video IDs and Labels
        """
        all_videos_set = numpy.take(all_video_ids, all_videos_indices_sorted_by_class, axis=0)
        all_labels_set = numpy.take(all_video_labels, all_videos_indices_sorted_by_class, axis=0)

        if store_to_db:
            dataset_dict = {}
            dataset_dict['metadata_type'] = "train_test_sets"

            # Create the Videos Set dictionary
            dataset_dict['videos_set'] = list()
            for i in range(0, len(all_videos_set)):
                video = {}
                video['video_id'] = all_video_ids[i]
                dataset_dict['videos_set'].append(video)

            # Create Labels Set dictionary
            dataset_dict['labels_set'] = list()
            for j in range(0, len(all_labels_set)):
                dataset_dict['labels_set'].append(all_labels_set[j])

            # Create VIDEO_IDS<->LABELS mappings if requested
            if include_video_ids_in_labels:
                dataset_dict['video_ids_with_labels'] = list()

                for k in range(0, len(all_videos_set)):
                    video_info = {}
                    video_info['video_id'] = all_videos_set[k]
                    video_info['label'] = all_labels_set[k]
                    dataset_dict['video_ids_with_labels'].append(video_info)

            # Store some stats also
            if self.nb_classes == 2:
                dataset_dict['dataset_stats'] = {}
                dataset_dict['dataset_stats']['appropriate'] = len(indices_appropriate)
                dataset_dict['dataset_stats']['inappropriate'] = len(indices_disturbing)
            elif self.nb_classes == 4:
                dataset_dict['dataset_stats'] = {}
                dataset_dict['dataset_stats']['appropriate'] = len(indices_appropriate)
                dataset_dict['dataset_stats']['disturbing'] = len(indices_disturbing)
                dataset_dict['dataset_stats']['inappropriate'] = len(indices_inappropriate)
                dataset_dict['dataset_stats']['other-irrelevant'] = len(indices_other)

            # Set the number of classes
            dataset_dict['nb_classes'] = self.nb_classes

            # Insert Train_Test Set dictionary in MongoDB
            self.dataset_metadata_col.insert(dataset_dict)

        print('--- Finished the generation of Dataset Videos and Labels set')

        return

    """ 
    Method that given a class as a string, return its number in the classes
    list. The is in one_hot shape. This method is also able to combine classes 
    based on the number of classes that we have set when initializing a Dataset
    object (nb_classes).
    We have the following classes for our problem:
        0 => appropriate
        1 => disturbing
        2 => inappropriate
        3 => other-irrelevant
    """
    def get_class_one_hot(self, class_str):
        # Encode it first.
        if class_str is not None:
            if self.nb_classes == 2:
                if class_str == 'appropriate' or class_str == 'other-irrelevant':
                    label_encoded = 0
                elif class_str == 'disturbing' or class_str == 'inappropriate':
                    label_encoded = 1
            elif self.nb_classes == 4:
                label_encoded = self.classes.index(class_str)
        else:
            # Handle the situation where the label is None
            if self.nb_classes == 2:
                label_encoded = 0 # set it as appropriate
            elif self.nb_classes == 4:
                label_encoded = 3 # set it as other-irrelevant

        # Now one-hot it. e.g., to_categorical(inappropriate = 2) => [0, 0, 1, 0]
        label_hot = to_categorical(label_encoded, len(self.classes))

        assert len(label_hot) == len(self.classes)

        return label_hot

    """ 
    Method that given a class as a string, return its number in the classes
    list. This method is also able to combine classes based on the number of classes
    that we have set when initializing a Dataset object (nb_classes).
    We have the following classes for our problem:
        0 => appropriate
        1 => disturbing
        2 => inappropriate
        3 => other-irrelevant
    """
    def get_class_to_categorical(self, class_str):
        # Encode it first.
        if class_str is not None:
            if self.nb_classes == 2:
                if class_str == 'appropriate' or class_str == 'other-irrelevant':
                    label_to_categorical = 0
                elif class_str == 'disturbing' or class_str == 'inappropriate':
                    label_to_categorical = 1
            elif self.nb_classes == 4:
                label_to_categorical = self.classes.index(class_str)
        else:
            # Handle the situation where the label is None
            if self.nb_classes == 2:
                label_to_categorical = 0  # set it as appropriate
            elif self.nb_classes == 4:
                label_to_categorical = 3  # set it as other-irrelevant
        return label_to_categorical

    """
    Method that returns the filename of the thumbnail of a given video_id.
    """
    def get_thumbnail_filename(self, video_id):
        """
        Create the absolute path to the directory where the thumbnail image of that video is stored
        """
        # Check in the POSSIBLY DISTURBING Videos' Thumbnails Directory
        thumbnail_filename = self.DATASET_DATA_BASE_DIR + self.THUMBNAILS_DIR + video_id + '/' + video_id + '.jpg'
        if os.path.isfile(thumbnail_filename):
            return thumbnail_filename

        # # Check in the RANDOM Videos' Thumbnails Directory
        # thumbnail_filename = self.DATASET_DATA_BASE_DIR + self.RANDOM_VIDEOS_BASE_DIR + self.THUMBNAILS_DIR + video_id + '/' + video_id + '.jpg'
        # if os.path.isfile(thumbnail_filename):
        #     return thumbnail_filename
        #
        # # Check in the CHILD RELATED Videos' Thumbnails Directory
        # thumbnail_filename = self.DATASET_DATA_BASE_DIR + self.CHILDRELATED_VIDEOS_BASE_DIR + self.THUMBNAILS_DIR + video_id + '/' + video_id + '.jpg'
        # if os.path.isfile(thumbnail_filename):
        #     return thumbnail_filename
        return None

    """
    Method that is called before starting the training in order to retrieve all thumbnail extracted
    features into a numpy array which actually is the memory of the server. This method is called for
    Cross Validation purposes.
    """
    def get_thumbnail_features_for_crossvalidation(self):
        """
        Get the saved extracted thumbnail features for the given Video ID
        """
        # construct absolute path to the stored file
        all_thumbnails_features_store_path = self.THUMBNAILS_FEATURES_DIR
        all_thumbnails_features_filename = all_thumbnails_features_store_path + 'all_thumbnails_features.npy'

        # Check if file exists and load its contents to a numpy array. If it does not exists then return None.
        if os.path.isfile(all_thumbnails_features_filename):
            return numpy.load(all_thumbnails_features_filename)
        else:
            return None

    """
    Method that retrieves the thumbnail's extracted features of a SINGLE VIDEO from the disk
    """
    def get_thumbnail_extracted_features(self, set_type, video_id):
        """
        Get the saved extracted thumbnail features for the given Video ID
        """
        # construct absolute path to the stored file
        thumbnail_features_store_path = self.THUMBNAILS_FEATURES_DIR + set_type + '/' + video_id
        thumbnail_features_store_path_filename = thumbnail_features_store_path + '/' + video_id + '_features' + '.npy'

        # Check if file exists and load its contents to a numpy array. If it does not exists then return None.
        if os.path.isfile(thumbnail_features_store_path_filename):
            return numpy.load(thumbnail_features_store_path_filename)
        else:
            return None

    """
    Method that checks if a (nested) key exists in a dict element
    """
    @staticmethod
    def key_exists(element, *keys):
        '''
        Check if *keys (nested) exists in `element` (dict).
        '''
        if type(element) is not dict:
            raise AttributeError('keys_exists() expects dict as first argument.')
        if len(keys) == 0:
            raise AttributeError('keys_exists() expects at least two arguments, one given.')

        _element = element
        for key in keys:
            try:
                _element = _element[key]
            except KeyError:
                return False
        return True

    """
    Method that receives as input the absolute path of a file and some data (numpy array) and
    it creates the file and all the missing subdirectories, and then it uses pickle to dump the 
    provided data into that file
    """
    @staticmethod
    def create_pickle_file_and_write_data(filename, data, protocol):
        # create the file as well as all the directories in the provided path that do not exist
        # os.makedirs(os.path.dirname(filename), exist_ok=True)
        # os.makedirs(os.path.dirname(filename))

        # write the provided data (bytes) in the created file
        pickle.dump(data, open(filename, 'wb'), protocol=protocol)

    """
    Method that returns the Headline of each video in the given corpus
    """
    def get_headlines_raw(self, num_examples=None, corpus="all", retrieved_videos_ids=None):
        print('--- Retrieving Video Headlines for corpus=' + corpus + '...')

        # Declare variables
        videos_in_sample = 0 # counter to use if num_examples value provided
        headlines = []

        # Get video ids for the given corpus
        if retrieved_videos_ids is None:
            if corpus == 'all':
                retrieved_videos_ids = self.get_dataset_video_ids(corpus=corpus, ignore_videos_with_disagreements=True)
            elif corpus == 'all_crossvalidation':
                retrieved_videos_ids = self.videos_set

        for video_id in retrieved_videos_ids:
            # Get video details
            video = self.videos_col.find_one({'id': video_id}, {'id': 1, 'snippet': 1})

            # get Video Title, encode unicode characters and append to the headlines array
            headline = self.encode_unicode_strings(video['snippet']['title'], isArray=False)
            headlines.append(headline)


            # check if there is a restriction in the number of sample
            if num_examples:
                videos_in_sample += 1
                if videos_in_sample >= num_examples:
                    break
        return headlines

    """
    Method that calculates the vocabulary size and max number of words in a headline for videos in the given coprus
    """
    def get_headlines_vocab_size_and_max_word_length(self, corpus='all', store_to_db=False):
        print('\n--- Started calculating headline vocabulary size and number of words in a headline for corpus=' + corpus + '...')

        # Check if we have already calculated this information
        details = self.dataset_metadata_col.find_one({'metadata_type': 'headlines_embeddings_info'})
        if details is not None:
            return details['vocab_size'], details['max_headlines_word_length']

        # init variables
        vocab_words = []
        vocab_size = 0
        max_headline_word_length = 0

        # Get video headlines for the given corpus
        all_video_headlines = self.get_headlines_raw(corpus=corpus)

        # Init progress bar
        progressBar = tqdm(total=len(all_video_headlines))
        for headline in all_video_headlines:
            # estimate the size of the vocab size of the current headline
            headline_words = text_to_word_sequence(headline)

            # Add each word to the vocabulary of words
            for word in headline_words:
                vocab_words.append(word)

            # Check if current headline word counter is the maximum that we have seen so far
            max_headline_word_length = max(len(headline_words), max_headline_word_length)

            # update progress bar
            progressBar.update(1)
        # close progress bar
        progressBar.close()

        # Find vocabulary size
        vocab_words = set(vocab_words)
        vocab_size = len(vocab_words)
        print('[HEADLINES] Vocabulary size: ' + str(vocab_size) + ', Max word length: ' + str(max_headline_word_length))

        if store_to_db:
            data_to_store = {}
            data_to_store['vocab_size'] = vocab_size
            data_to_store['max_headlines_word_length'] = max_headline_word_length
            self.dataset_metadata_col.update_one({'metadata_type': 'headlines_embeddings_info'}, {"$set": data_to_store}, upsert=True)
        return vocab_size, max_headline_word_length

    """
    Method that performs the required pre-processing for all the headlines for the given corpus
    using keras.preprocessing.text.one_hot. It is actually a bag-of-words technique.
    """
    def preprocess_headlines_one_hot(self, vocab_size, headline_max_words_length, corpus='all'):
        print('')
        print('--- Started headline pre-processing for corpus=' + corpus + ' using Keras one_hot...')

        # Get video headlines for the given corpus
        all_video_headlines = self.get_headlines_raw(corpus=corpus)

        # Double the size of the vocabulary to minimize collisions when hashing words. BUT if we do that then remember to
        # also change the first parameter of the Embedding layer
        _vocab_size = vocab_size

        # Integer encode the document
        encoded_headlines = [one_hot(headline, _vocab_size) for headline in all_video_headlines]

        # Perform padding
        headline_features = sequence.pad_sequences(encoded_headlines, maxlen=headline_max_words_length)
        return numpy.array(headline_features)

    """
    Method that retrieves the preprocessed video headlines features from the disk for 
    the given set (TRAIN or TEST)
    """
    def get_headlines_features_for_training(self, set_type, embeddings_features=True, is_crossvalidation=False, isSimpleNN=False):
        # Create the absolute path to the video statistics features file
        if isSimpleNN:
            filename = self.HEADLINES_FEATURES_BASE_DIR + 'all_headlines_preprocessed.p'
        else:
            if embeddings_features:
                if is_crossvalidation:
                    filename = self.HEADLINES_FEATURES_BASE_DIR + 'all_headlines_features.p'
                else:
                    filename = self.HEADLINES_FEATURES_BASE_DIR + set_type + '/video_headlines_features_' + set_type + '_set.p'
            else:
                filename = self.HEADLINES_FEATURES_BASE_DIR + set_type + '/video_headlines_features_' + set_type + '_set1000.p'

        # Retrieve the video headlines features from the disk as a numpy array
        preprocessed_video_headlines = pickle.load(open(filename, 'rb'))
        return preprocessed_video_headlines

    """
    Method that returns the VIDEO TAGS of each video in the given corpus
    """
    def get_video_tags_raw(self, num_examples=None, corpus="all"):
        print('--- Retrieving Video Tags for corpus=' + corpus + '...')

        # Declare variables
        videos_in_sample = 0 # counter to use if num_examples value provided
        all_video_tags, empty_array = [], []

        # Get video ids for the given corpus
        if corpus == 'all':
            retrieved_videos_ids = self.get_dataset_video_ids(corpus=corpus, ignore_videos_with_disagreements=True)
        elif corpus == 'all_crossvalidation':
            retrieved_videos_ids = self.videos_set

        for video_id in retrieved_videos_ids:
            # Get video details
            video = self.videos_col.find_one({'id': video_id}, {'id': 1, 'snippet': 1})

            # Get Video Tags if there are any
            if self.key_exists(video, 'snippet', 'tags'):
                curr_video_tags = self.encode_unicode_strings(video['snippet']['tags'], isArray=True)

                # Make sure that all tags are words and not phrases
                final_curr_video_tags = []
                for tag in curr_video_tags:
                    tag_words = text_to_word_sequence(tag)
                    final_curr_video_tags += tag_words

                # Append current video tags to the result array with all the video tags
                all_video_tags.append(final_curr_video_tags) # tags is an array
            else:
                # This video has not tags so append an empty array
                all_video_tags.append(empty_array)

            # check if there is a restriction in the number of sample
            if num_examples:
                videos_in_sample += 1
                if videos_in_sample >= num_examples:
                    break

        return all_video_tags

    """
    Method that calculates the vocabulary size and the max number of tags in videos in the given coprus
    """
    def get_video_tags_vocab_size_and_max_count(self, corpus='all', store_to_db=False):
        print('\n--- Started calculating video tags vocabulary size and max number of tags in a video for corpus=' + corpus + '...')

        # Check if they already exists in MongoDB
        details = self.dataset_metadata_col.find_one({'metadata_type': 'video_tags_embeddings_info'})
        if details is not None:
            return details['vocab_size'], details['max_video_tags_length']

        # init variables
        vocab_words = list()
        max_video_tags_length = 0

        # Get video headlines for the given corpus
        all_videos_tags = self.get_video_tags_raw(corpus=corpus)

        # Init progress bar
        progressBar = tqdm(total=len(all_videos_tags))

        # Iterate through each video tags and calculate the max number of tags in a video
        # and also create a dictionary with all the unique tags and also calculate the vocabulary
        # size which is the number of unique tags in all videos for the given corpus
        for video_tags in all_videos_tags:
            # check if the current video has any tags
            if len(video_tags) == 0:
                # update progress bar
                progressBar.update(1)

                continue # skip that video since it has not tags

            # iterate through each tag of a video
            for tag in video_tags:
                vocab_words.append(tag)

            # Check if current headline word counter is the maximum that we have seen so far
            max_video_tags_length = max(len(video_tags), max_video_tags_length)

            # update progress bar
            progressBar.update(1)

        # Find vocabulary size
        vocab_words = set(vocab_words)
        vocab_size = len(vocab_words)

        progressBar.close()
        print('[VIDEO_TAGS] Vocabulary size: ' + str(vocab_size) + ', Max video tags length: ' + str(max_video_tags_length))

        if store_to_db:
            data_to_store = {}
            data_to_store['vocab_size'] = vocab_size
            data_to_store['max_video_tags_length'] = max_video_tags_length
            self.dataset_metadata_col.update_one({'metadata_type': 'video_tags_embeddings_info'}, {"$set": data_to_store}, upsert=True)
        return vocab_size, max_video_tags_length

    """
    Method that performs the required pre-processing for the video tags of all the videos for the given corpus
    using keras.preprocessing.text.one_hot. It is actually a bag-of-words technique.
    """
    def preprocess_video_tags_one_hot(self, vocab_size, video_tags_max_length, corpus='all'):
        print('')
        print('--- Started video tags pre-processing for corpus=' + corpus + ' using Keras one_hot...')

        # Get video headlines for the given corpus
        all_videos_tags = self.get_video_tags_raw(corpus=corpus)

        # init variables
        curr_encoded_video_tags, encoded_all_video_tags = list(), list()

        # Double the size of the vocabulary to minimize collisions when hashing words. BUT if we do that then remember to
        # also change the first parameter of the Embedding layer
        # _vocab_size = round(vocab_size * 2)
        _vocab_size = vocab_size

        progressBar = tqdm(total=len(all_videos_tags))
        # Integer encode the document
        for video_tags in all_videos_tags:
            # re-init current video encoded video tags array
            curr_encoded_video_tags = []

            if len(video_tags) == 0:
                # this video has not tags so add an empty array
                encoded_all_video_tags.append(curr_encoded_video_tags)
            else:
                # encode each tag of the video separately
                for tag in video_tags:
                    curr_encoded_video_tags += one_hot(tag, _vocab_size)
                    # curr_encoded_video_tags.append(one_hot(tag, _vocab_size))

                # append the current video's encoded video tags array
                encoded_all_video_tags.append(curr_encoded_video_tags)

            # Update the progress bar
            progressBar.update(1)

        # Finish the progress bar
        progressBar.close()

        # Perform padding
        video_tags_features = sequence.pad_sequences(encoded_all_video_tags, maxlen=video_tags_max_length)
        return numpy.array(video_tags_features)

    """
    Method that retrieves the preprocessed video tags features from the disk for the given set (TRAIN or TEST)
    """
    def get_video_tags_features_for_training(self, set_type, embeddings_features=True, is_crossvalidation=False):
        # Create the absolute path to the video statistics features file
        if embeddings_features:
            if is_crossvalidation:
                filename = self.VIDEO_TAGS_FEATURES_BASE_DIR + 'all_video_tags_features.p'
            else:
                filename = self.VIDEO_TAGS_FEATURES_BASE_DIR + set_type + '/video_tags_features_' + set_type + '_set.p'
        else:
            filename = self.VIDEO_TAGS_FEATURES_BASE_DIR + set_type + '/video_tags_features_' + set_type + '_set100.p'

        # Retrieve the video headlines features from the disk as a numpy array
        preprocessed_video_headlines = pickle.load(open(filename, 'rb'))

        return preprocessed_video_headlines

    """
    Method that retrieves from MongoDB the following embedding information for the given type:
    1. vocabulary size
    2. vocabulary words
    3. max sequence length
    """
    def get_embeddings_information_from_db(self, embedding_type_info):
        print('--- Retrieving embedding information for ' + embedding_type_info + '...')

        # retrieve embedding info from MongoDB for the given type
        retrieved_embedding_information = self.dataset_metadata_col.find_one({'metadata_type': embedding_type_info})

        if embedding_type_info == 'headlines_embeddings_info':
            vocab_size = retrieved_embedding_information['vocab_size']
            max_sequence_length = retrieved_embedding_information['max_headlines_word_length']
        elif embedding_type_info == 'video_tags_embeddings_info':
            vocab_size = retrieved_embedding_information['vocab_size']
            max_sequence_length = retrieved_embedding_information['max_video_tags_length']
        else:
            raise Exception('Cannot find stored Embedding information. Did you generate them?')

        return vocab_size, max_sequence_length

    """
    Method that accepts either a String or an Array, encodes all unicode characters in the string or array
    and returns the encoded result
    """
    @staticmethod
    def encode_unicode_strings(input_data, isArray=True):
        if isArray:
            encoded_array = []

            for item in input_data:
                encoded_array.append(unicodedata.normalize('NFKD', item).encode('ascii', 'ignore').decode("utf-8", "strict"))
                # encoded_array.append(unicodedata.normalize('NFKD', item).encode('utf-8', 'ignore'))
            return encoded_array
        else:
            encoded_string = unicodedata.normalize('NFKD', input_data).encode('ascii', 'ignore').decode("utf-8", "strict")
            # encoded_string = unicodedata.normalize('NFKD', input_data).encode('utf-8', 'ignore')
            return encoded_string

    """
    Method that calculates the number of emoticons in the given text
    """
    @staticmethod
    def get_num_of_emoticons(str):
        num_emoticons = 0

        for character in str:
            if character in emoji.UNICODE_EMOJI:
                num_emoticons += 1

        return num_emoticons

    """
    Method that calculates the Jaccard Similarity between two strings
    """
    @staticmethod
    def get_jaccard_similarity(str1, str2):
        a = set(str1.split())
        b = set(str2.split())
        c = a.intersection(b)
        jaccard_sim = float(len(c)) / (len(a) + len(b) - len(c))

        return float("{0:.2f}".format(jaccard_sim))

    """
    Method that converts a YouTube video's duration to seconds
    """
    def YTDurationToSeconds(self, duration):
        match = re.match('PT(\d+H)?(\d+M)?(\d+S)?', duration).groups()
        hours = self._js_parseInt(match[0]) if match[0] else 0
        minutes = self._js_parseInt(match[1]) if match[1] else 0
        seconds = self._js_parseInt(match[2]) if match[2] else 0
        return hours * 3600 + minutes * 60 + seconds

    """
    JavaScript-like parseInt method
    See also: # https://gist.github.com/douglasmiranda/2174255
    """
    @staticmethod
    def _js_parseInt(string):
        return int(''.join([x for x in string if x.isdigit()]))

    """
    Method that shuffles two arrays in the same exact way
    """
    @staticmethod
    def suffle_arrays_similarly(list1, list2, list3=None):
        indices = numpy.arange(list1.shape[0])
        numpy.random.shuffle(indices)
        list1 = list1[indices]
        list2 = list2[indices]
        if list3 is not None:
            list3 = list3[indices]
            return list1, list2, list3
        else:
            return list1, list2

    """
    Method that shuffles two arrays in the same exact way
    """
    @staticmethod
    def sort_arrays_similarly(list1, list2):
        return (list(t) for t in zip(*sorted(zip(list1, list2))))

    """
    Method that the takes the labels of a X Set and returns the indices of a TRAIN and a VALIDATION Set
    """
    def stratified_train_val_split(self, nb_classes, set_labels, val_size=0.2):
        # print('Splitting to TRAIN and VAL set. TOTAL Videos: %d' % (len(set_labels)))
        # Declare vaiables
        indices_train, indices_val = list(), list()

        if nb_classes == 2:
            # Get the indices of the Appropriate videos
            indices_appropriate = [i for i, x in enumerate(set_labels) if x == 0.0]

            # Get the indices of the Disturbing videos
            indices_disturbing = [i for i, x in enumerate(set_labels) if x == 1.0]

            #
            # Create the TRAIN and the VAL Sets
            #
            # APPROPRIATE
            total_appropriate_train = int(len(indices_appropriate) * (1 - val_size))
            indices_train = indices_appropriate[0:total_appropriate_train]
            indices_val = indices_appropriate[total_appropriate_train:len(indices_appropriate)]

            # DISTURBING
            total_disturbing_train = int(len(indices_disturbing) * (1 - val_size))
            indices_train += indices_disturbing[0:total_disturbing_train]
            indices_val += indices_disturbing[total_disturbing_train:len(indices_disturbing)]

        elif nb_classes == 4:
            # Get the indices of the Appropriate videos
            indices_appropriate = [i for i, x in enumerate(set_labels) if x == 0.0]

            # Get the indices of the Disturbing videos
            indices_disturbing = [i for i, x in enumerate(set_labels) if x == 1.0]

            # Get the indices of the Inappropriate videos
            indices_inappropriate = [i for i, x in enumerate(set_labels) if x == 2.0]

            # Get the indices of the Other-Irrelevant videos
            indices_other = [i for i, x in enumerate(set_labels) if x == 3.0]

            #
            # Create the TRAIN and the VAL Sets
            #
            # APPROPRIATE
            total_appropriate_train = int(len(indices_appropriate) * (1 - val_size))
            indices_train = indices_appropriate[0:total_appropriate_train]
            indices_val = indices_appropriate[total_appropriate_train:len(indices_appropriate)]

            # DISTURBING
            total_disturbing_train = int(len(indices_disturbing) * (1 - val_size))
            indices_train += indices_disturbing[0:total_disturbing_train]
            indices_val += indices_disturbing[total_disturbing_train:len(indices_disturbing)]

            # INAPPROPRIATE
            total_inappropriate_train = int(len(indices_inappropriate) * (1 - val_size))
            indices_train += indices_inappropriate[0:total_inappropriate_train]
            indices_val += indices_inappropriate[total_inappropriate_train:len(indices_inappropriate)]

            # OTHER-IRRELEVANT
            total_other_train = int(len(indices_other) * (1 - val_size))
            indices_train += indices_other[0:total_other_train]
            indices_val += indices_other[total_other_train:len(indices_other)]

        print('--- [TRAIN_VAL_SPLIT] TOTAL VIDEOS: %d | TOTAL TRAIN: %d, TOTAL VAL: %d' % (len(set_labels), len(indices_train), len(indices_val)))
        return indices_train, indices_val