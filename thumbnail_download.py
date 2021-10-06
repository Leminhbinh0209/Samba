import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import requests
from meta_helper.utils import  CNN_MODEL
from easydict import EasyDict as edict
import yaml

def main(config):
    uscs_dir = f"{config.data_folder}/{config.dataset}_data/" 
    uscs_big_meta = pd.read_csv(uscs_dir + f"meta_data/{config.dataset.lower()}_big_meta.csv", index_col='video_id', lineterminator='\n').reset_index()

    img_dir = uscs_dir + f"meta_data/thumbnail/"
    emb_dir = uscs_dir + f"meta_data/thumbnail_emb/"

    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(emb_dir, exist_ok=True)

    list_thumbnail = uscs_big_meta.thumbnail_url.values.tolist() 
    list_ids = uscs_big_meta.video_id.values.tolist()

    for vid, url_vid in tqdm(zip(list_ids, list_thumbnail)):
        if not os.path.exists(img_dir + vid + '.png'):
            img_data  = requests.get(url_vid).content
            with open(img_dir + vid + '.png', 'wb') as handler:
                handler.write(img_data)
    print("Fininsh download\nStart embedding thumbnail image using InceptionNet")

    CNN_feature_extractor_model = CNN_MODEL()
    for video_id in tqdm(list_ids):
        if not os.path.exists(emb_dir  + video_id + '.npy'):
            thumbnail_features = CNN_feature_extractor_model.extract_features_image(frame_image_path=img_dir + video_id + '.png')
            with open(emb_dir  + video_id + '.npy', 'wb') as f:
                np.save(f, thumbnail_features)
    print("Fininsh embedding")    

if __name__ == "__main__":
    with open('./config/config_pre.yaml', 'r') as stream:
        config = yaml.safe_load(stream)
    config = edict(config)
    
    main(config)