import sys
import pandas as pd
import pickle
from easydict import EasyDict as edict
import yaml
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

def main(config):
    uscs_dir = f"{config.data_folder}/{config.dataset}_data/" 
    uscs_big_meta = pd.read_csv(uscs_dir + f"meta_data/{config.dataset.lower()}_big_meta.csv", index_col='video_id', lineterminator='\n').reset_index()

    print("Start split dataset")
    channel_id_all = uscs_big_meta.channel_id.unique()
    while True:
        train_val_set_indices, test_set_indices= train_test_split(np.arange(len(channel_id_all)), test_size=0.2, random_state=np.random.randint(100000))
        train_channels = channel_id_all[train_val_set_indices]
        test_channels = channel_id_all[test_set_indices]
        _, val_channels = train_test_split(train_channels, random_state=41, train_size=0.8)

        train_df  = uscs_big_meta.loc[uscs_big_meta['channel_id'].isin(train_channels)]
        test_df = uscs_big_meta.loc[uscs_big_meta['channel_id'].isin(test_channels)]
        train_index = train_df.index.values

        test_index = test_df.index.values
        train_df.reset_index(drop=True, inplace=True)
        val_index = train_df.loc[train_df['channel_id'].isin(val_channels)].index.values

        train_ratio = train_df.onehot_label.values.sum() / len(train_df.onehot_label.values)
        test_ratio = test_df.onehot_label.values.sum() / len(test_df.onehot_label.values)
        
        print("Train positive ratio: {:.4f}".format(train_ratio))
        print("Test positive ratio: {:.4f}".format(test_ratio))
        if not np.abs(train_ratio-test_ratio) < 0.05:
            continue
        else:
            break
        
    print("Save training/test video index")
    textfile = open(f"{uscs_dir}train_videos.txt", "w")
    for element in train_df.video_id:
        textfile.write(element + "\n")
    textfile.close()
    textfile = open(f"{uscs_dir}test_videos.txt", "w")
    for element in test_df.video_id:
        textfile.write(element + "\n")
    textfile.close()

if __name__ == "__main__":
    print("This process will re-generate train and test set.\nPLEASE MAKE SURE YOU WANT TO CHANGE BY REMOVING sys.exit()")
    # sys.exit()
    with open('./config/config_pre.yaml', 'r') as stream:
        config = yaml.safe_load(stream)
    config = edict(config)
    main(config)