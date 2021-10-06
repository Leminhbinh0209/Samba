# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 20:33:33 2018

@author: kostantinos.papadamou

This Class implements the methods that generates extracted features for each video,
which we will then use to train our LSTM. In addition, it also generates the features
for each video thumbnail that we will pass later to our Fusing Network.

We actually take all frames of each video and we pass them to a pre-trained CNN (Inception V3)
which extracts features. We do that by removing the last softmax layer  of the pre-trained CNN
which makes the classification. By doing that the CNN actually outputs a 1x2048 vector that
includes the features of each frame.

When running this by looping through all videos in our corpus we end up with a numpy array of
X * Y dimensions where each row X of that array is the features vector with Y length (1x1x2048 - 2048d)
of each frame of the video and this numpy array is stored in a file on the disk where we will
end up to have one file per video.
"""
import os
# Restrict the script to run everything on the first GPU [GPU_ID=0]
os.environ ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Import Keras Tensoflow Backend
from keras import backend as K
import tensorflow as tf

import numpy as np
import os.path
from tqdm import tqdm
from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model, load_model


class CNN_MODEL(object):
    def __init__(self, dataset, weights=None):
        """
        Either load pre-trained weights from ImageNet (None), or load our saved weights from our own training.
        """

        # Get Dataset Object
        self.dataset = dataset

        # Get weights type
        self.weights = weights

        # Set image input shape
        self.image_input_shape_channels = (299, 299, 3)
        self.image_input_shape = (299, 299)

        # Check whether Keras is using GPU
        # NOTE: If it returns an empty array it means that Keras is not using GPU
        print('')
        print('--- Available GPUs: ' + str(K.tensorflow_backend._get_available_gpus()))
        print('')

        if weights is None:
            # Get CNN model with pre-trained ImageNet weights. Include also the top fully connected layer
            # and we will freeze the last softmax classification layer in the next line
            CNN_model = InceptionV3(weights='imagenet', include_top=True)

            # Freeze the last softmax layer. We'll extract features at the final pool layer.
            self.model = Model(inputs=CNN_model.input, outputs=CNN_model.get_layer('avg_pool').output)

        else:
            # Load the model first.
            self.model = load_model(weights)

            # Then remove the top so we get features not predictions.
            # From: https://github.com/fchollet/keras/issues/2371
            self.model.layers.pop()
            self.model.layers.pop()  # two pops to get to pool layer
            self.model.outputs = [self.model.layers[-1].output]
            self.model.output_layers = [self.model.layers[-1]]
            self.model.layers[-1].outbound_nodes = []


    """
    Method that takes as input the path of an image (in our case either a VIDEO FRAME, or a video THUMBNAIL) 
    converts it to a numpy array using Keras.preprocessing package and then extracts the features from that 
    image frame by running it into the pre-trained CNN Inception V3 model
    """
    def extract_features_image(self, frame_image_path, isEmptyImage=False):
        # Load the image from the jpg file with keras.preprocessing package
        # We load the image in a (299x299x3) size because this is what Inception V3 CNN accepts
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

        if self.weights is None:
            # For imagenet/default network:
            features = features[0]
        else:
            # For loaded network:
            features = features[0]

        return features

    """
    Method that extracts the features of the thumbnail of each video in our Dataset that will be then used
    for Cross Validation. It uses the extract_features_image to extract thumbnail's features.
    """
    def extract_thumbnails_features_crossvalidation(self, all_video_ids=None, store_all_to_single_file=False):
        print('--- Started extracting THUMBNAIL features for all videos in the for Cross-Validation...')
        skipped_thumbnails_cntr = 0

        # Declare and create the directory where all the features of all the THUMBNAILS will be stored
        all_thumbnails_features = list()
        all_thumbnails_features_store_path = self.dataset.THUMBNAILS_FEATURES_DIR
        all_thumbnails_features_filename = all_thumbnails_features_store_path + 'all_thumbnails_features'

        # Get all video_ids in the given corpus
        if not all_video_ids:
            all_video_ids = self.dataset.videos_set

        # Create a Progress Bar
        progressBar = tqdm(total=len(all_video_ids))
        # Iterate through the videos and generate the features for each Video's Thumbnail
        for video_id in all_video_ids:
            # Create thumbnail features path and filename
            thumbnail_features_store_path = self.dataset.THUMBNAILS_FEATURES_DIR + video_id
            thumbnail_features_store_path_filename = thumbnail_features_store_path + '/' + video_id + '_features'

            # Check whether we already have the features for that specific video
            if os.path.isfile(thumbnail_features_store_path_filename + '.npy') and not store_all_to_single_file:
                progressBar.update(1)
                skipped_thumbnails_cntr += 1
                continue # SKIP that video

            # Get the Actual Thumbnail image filename
            thumbnail_image_filename = self.dataset.get_thumbnail_filename(video_id)

            """
            Verify that we received a correct filename and extract the features of the thumbnail
            image using the pre-trained CNN Inception V3 Model
            """
            if thumbnail_image_filename is not None:
                thumbnail_features = self.extract_features_image(thumbnail_image_filename)
            else:
                # thumbnail does not exits so create an empty features array
                thumbnail_features = self.extract_features_image("", isEmptyImage=True)
                skipped_thumbnails_cntr += 1


            """
            Create directory to store the video's thumbnail features file
            """
            if not store_all_to_single_file:
                original_umask = os.umask(0)
                try:
                    # Create Thumbnail Features Sequence Base Directory
                    if not os.path.exists(thumbnail_features_store_path):
                        os.makedirs(thumbnail_features_store_path, 0o777)
                finally:
                    os.umask(original_umask)
                # Store the features sequence of that video to disk
                np.save(thumbnail_features_store_path_filename, thumbnail_features)
            else:
                all_thumbnails_features.append(thumbnail_features)

            # Update Progress Bar
            progressBar.update(1)
        # Finish progress bar
        progressBar.close()

        # Store all Thumbnail Features array into a file
        if store_all_to_single_file:
            # Create Thumbnail Features Directory
            original_umask = os.umask(0)
            try:
                if not os.path.exists(all_thumbnails_features_store_path):
                    os.makedirs(all_thumbnails_features_store_path, 0o777)
            finally:
                os.umask(original_umask)
            # Store all Thumbnails Features them to a file
            np.save(all_thumbnails_features_filename, all_thumbnails_features)

        print('--- SKIPPED VIDEOS: %d' % (skipped_thumbnails_cntr))
        return
