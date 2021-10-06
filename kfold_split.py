import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import pickle
from easydict import EasyDict as edict
import yaml
def main(config):
    uscs_dir = f"{config.data_folder}/{config.dataset}_data/" 
    uscs_big_meta = pd.read_csv(uscs_dir + f"meta_data/{config.dataset.lower()}_big_meta.csv", index_col='video_id', lineterminator='\n').reset_index()

    print("Start split dataset")
    k_fold_dict = dict()
    channel_id_all = uscs_big_meta.channel_id.unique()
    stratified_kfold = KFold(n_splits=5, shuffle=True, random_state=config.random_state)
    for id, (train_val_set_indices, test_set_indices) in enumerate(stratified_kfold.split(channel_id_all)):
        train_channels = channel_id_all[train_val_set_indices]
        test_channels = channel_id_all[test_set_indices]
        _, val_channels = train_test_split(train_channels, random_state=41, train_size=0.8)
        
        train_df  = uscs_big_meta.loc[uscs_big_meta['channel_id'].isin(train_channels)]
        test_df = uscs_big_meta.loc[uscs_big_meta['channel_id'].isin(test_channels)]
        train_index = train_df.index.values
        
        test_index = test_df.index.values
        train_df.reset_index(drop=True, inplace=True)
        val_index = train_df.loc[train_df['channel_id'].isin(val_channels)].index.values
        
        k_fold_dict[id] = dict()
        k_fold_dict[id]['train'] = list(train_index)
        k_fold_dict[id]['val'] = list(val_index)
        k_fold_dict[id]['test'] = list(test_index)

    print(f"Finish... save the kfold in {config.dataset}_big_meta")
    with open(uscs_dir + "meta_data/k_fold_channel.json", 'wb') as fp:
        pickle.dump(k_fold_dict, fp)
    
if __name__ == "__main__":
    with open('./config/config_pre.yaml', 'r') as stream:
        config = yaml.safe_load(stream)
    config = edict(config)
    
    main(config)