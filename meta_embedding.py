import os
import pandas as pd
import json
import numpy as np
from tqdm import tqdm
from meta_helper.utils import *
from meta_helper.h5py_func import save_dict
from tensorflow.keras.preprocessing.text import  text_to_word_sequence
from easydict import EasyDict as edict
import yaml
import h5py


os.environ["CUDA_VISIBLE_DEVICES"] = '0' 

def main(config):
    uscs_dir = f"{config.data_folder}/{config.dataset}_data/" 
    uscs_big_meta = pd.read_csv(uscs_dir + f"meta_data/{config.dataset.lower()}_big_meta.csv", index_col='video_id', lineterminator='\n').reset_index()
    emb_folder = f"{uscs_dir}/meta_data/meta_emb/"
    os.makedirs(emb_folder, exist_ok=True)
    img_dir = uscs_dir + f"meta_data/thumbnail/"
    thumb_emb_dir = uscs_dir + f"meta_data/thumbnail_emb/"

    '''
        First round for statistic
    '''
    headlines_voc = dict()
    max_headlines = 0
    tags_voc = dict()
    max_tags = 0
    for index, video_information in tqdm(uscs_big_meta.iterrows()):
        # Headlines
        video_headline = video_information.title
        headline_sequence = text_to_word_sequence(video_headline)
        max_headlines = max(max_headlines, len(headline_sequence))
        for w in headline_sequence:
            headlines_voc[w] = 1 if w not in headlines_voc else headlines_voc[w] + 1 
            
        # Tags
        try:
            video_tags = video_information.tags
        except AttributeError:
            video_tags = []
        video_tags=eval(video_tags) if len(video_tags) > 1 else []
        for tag in video_tags:
            tags_voc[tag] = 1 if tag not in tags_voc else tags_voc[tag] + 1 
        max_tags = max(max_tags, len(video_tags))

    save_voc = {'headline':{'voc_size':len(headlines_voc), 'max':max_headlines},
            'tag':{'voc_size':len(tags_voc), 'max':max_tags},
            'subtitle_emb': 500}
    with open(emb_folder + 'voc.json', 'w') as outfile:
        json.dump(save_voc, outfile)

    '''
        Seconnd round for statistic
    '''
    X_emb_data =  []
    data_dict = {}
    CNN_feature_extractor_model = CNN_MODEL()
    for index, video_information in tqdm(uscs_big_meta.iterrows()):
        X_general_style_features_in = get_video_general_style_features(video_information)
        video_headline = video_information.title
        X_headline_features_in = preprocess_headlines_one_hot(video_headline=video_headline, 
                                                            vocab_size=len(headlines_voc), 
                                                            headline_max_words_length=max_headlines)
        try:
            video_tags = video_information.tags
        except AttributeError:
            video_tags = []
        X_video_tags_features_in = preprocess_video_tags_one_hot(video_tags=eval(video_tags) if len(video_tags) > 1 else [], 
                                                                vocab_size=len(tags_voc), 
                                                                video_tags_max_length=max_tags)

        if os.path.exists(thumb_emb_dir + video_information.video_id + '.npy'):
            with open(thumb_emb_dir + video_information.video_id + '.npy', 'rb') as f:
                thumbnail_features = np.load(f)
        else:
            thumbnail_features = CNN_feature_extractor_model.extract_features_image(frame_image_path=img_dir + video_information.video_id + '.png')
            with open(thumb_emb_dir + video_information.video_id + '.npy', 'wb') as f:
                np.save(f, thumbnail_features)
    
        data_dict[video_information.video_id] = {
            'thumbnail': thumbnail_features,
            'headline': X_headline_features_in[0],
            'style': X_general_style_features_in[0],
            'tags': X_video_tags_features_in[0],
            'y': video_information.onehot_label
        }
        

    save_dict(data_dict, emb_folder + 'meta_embedding.hdf5')
    

if __name__ == "__main__":
    with open('./config/config_pre.yaml', 'r') as stream:
        config = yaml.safe_load(stream)
    config = edict(config)
    
    main(config)
