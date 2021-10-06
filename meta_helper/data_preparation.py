# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 20:33:33 2018

@author: kostantinos.papadamou

This file executes all the required functions that do the pre-processing of our data
"""

import pickle
import Dataset
import CNN_model
import numpy
from sklearn import preprocessing

if __name__ == "__main__":
    NB_CLASSES = 2

    """
    DATASET Preparation
    """
    # TODO: RUN THIS FIRST to generated the dataset_metadata MongoDB collection with all the Videos and their labels
    #       that will be later used for Data Preprocessing and Model Training
    # Prepare dataset (Video IDs and LABELS) for all the Videos in our Dataset for K-Fold Cross Validation
    dataset = Dataset.DATASET(prepare_dataset_crossvalidation=False, nb_classes=NB_CLASSES)
    print('\n--- Started preparing DATASET and LABELS for K-Fold Cross Validation %d-classes...' % NB_CLASSES)
    dataset.prepare_dataset_for_crossvalidation(ignore_videos_with_disagreement=True,
                                                include_age_restricted_videos=False,  # we are not actually ignoring them but we only include the annotated ones
                                                include_annotated_age_restricted=True,
                                                store_to_db=True,
                                                include_video_ids_in_labels=True)

    """
    Re-initiate DATASET Object to preprocess input data
    """
    del dataset
    # Create Dataset Object
    dataset = Dataset.DATASET(prepare_dataset_crossvalidation=True, nb_classes=NB_CLASSES)

    """
    Create a CNN video feature extraction Object
    """
    CNN_feature_extractor_model = CNN_model.CNN_MODEL(dataset)

    """
    THUMBNAILS pre-processing
    """
    # Extracts the features for the thumbnail of each video in our GROUND TRUTH DATASET for CROSS-VALIDATION
    print('\n--- Started extracting features for all the thumbnails in our GROUND TRUTH for CROSS-VALIDATION...')
    CNN_feature_extractor_model.extract_thumbnails_features_crossvalidation(all_video_ids=dataset.videos_set, store_all_to_single_file=True)

    """
    STATISTICS pre-processing
    """
    # Pre-process the video statistics for the WHOLE DATASET for CROSS-VALIDATION
    print('\n--- Started pre-processing Video Statistics for the WHOLE DATASET for CROSS-VALIDATION...')
    video_statistics_features = dataset.preprocess_video_statistics(set_video_ids=dataset.videos_set)
    video_statistics_features = preprocessing.normalize(video_statistics_features, axis=0).astype("float32")
    # Create statistics features filename and write in it the pre-processed statistics features
    dataset.create_pickle_file_and_write_data(
        filename=dataset.STATISTICS_FEATURES_BASE_DIR + 'all_statistics_features.p',
        data=video_statistics_features,
        protocol=3
    )

    """
    HEADLINES pre-processing
    """
    # Get max number of words that a headline has and the vocabulary size in our WHOLE DATASET
    all_headlines_vocab_size, all_max_headline_word_length = dataset.get_headlines_vocab_size_and_max_word_length(
        corpus='all_crossvalidation',
        store_to_db=True
    )

    # Pre-process the Headlines for the WHOLE DATASET for CROSS-VALIDATION
    print('\n--- Started pre-processing HEADLINES for the WHOLE DATASET for CROSS-VALIDATION...')
    all_headlines_features = dataset.preprocess_headlines_one_hot(all_headlines_vocab_size, all_max_headline_word_length, corpus='all_crossvalidation')
    # Create headline features filename and write in it the pre-processed headlines features
    dataset.create_pickle_file_and_write_data(
        filename=dataset.HEADLINES_FEATURES_BASE_DIR + 'all_headlines_features.p',
        data=all_headlines_features,
        protocol=3
    )

    """
    VIDEO TAGS pre-processing
    """
    # Get max number of tags that a video has and the vocabulary size in our WHOLE DATASET
    all_video_tags_vocab_size, all_max_video_tags_count = dataset.get_video_tags_vocab_size_and_max_count(corpus='all_crossvalidation', store_to_db=True)

    # Pre-process the VIDEO TAGS for the WHOLE DATASET for CROSS-VALIDATION
    print('\n--- Started pre-processing VIDEO TAGS for the WHOLE DATASET for CROSS-VALIDATION...')
    all_video_tags_features = dataset.preprocess_video_tags_one_hot(
        vocab_size=all_video_tags_vocab_size,
        video_tags_max_length=all_max_video_tags_count,
        corpus='all_crossvalidation'
    )
    # Create Video Tags features filename and write in it the pre-processed Video Tags features
    dataset.create_pickle_file_and_write_data(
        filename=dataset.VIDEO_TAGS_FEATURES_BASE_DIR + 'all_video_tags_features.p',
        data=all_video_tags_features,
        protocol=3
    )
