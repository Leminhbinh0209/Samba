import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
from easydict import EasyDict as edict
import yaml
from nltk.tokenize import TweetTokenizer
import gensim
import codecs
from gensim.models.word2vec import Word2Vec
from gensim.models import word2vec
from time import time
from datetime import timedelta
twitter = TweetTokenizer()

def tokenizer_twitter_morphs(doc):
    return twitter.tokenize(doc)

def tokenizer_twitter_noun(doc): 
    return twitter.nouns(doc)

def tokenizer_twitter_pos(doc): 
    return twitter.pos(doc, norm=True, stem=True)

def get_features(words, model, num_features):
    # 출력 벡터 초기화
    feature_vector = np.zeros((num_features), dtype=np.float32)
    
    num_words = 0
    # 어휘 사전 준비
    index2word_set = set(model.wv.index_to_key)
    
    for w in words:
        if w in index2word_set:
            num_words = 1
            feature_vector += model.wv[w].reshape((num_features, ))
    feature_vector = np.divide(feature_vector, num_words)
    return feature_vector

def get_dataset(reviews, model, num_features):
    dataset = []
    type(reviews)
    for s in reviews:
        dataset.append(get_features(s, model, num_features))
    reviewFeatureVecs = np.stack(dataset)

    return reviewFeatureVecs
def main(config):
    uscs_dir = f"{config.data_folder}/{config.dataset}_data/" 
    uscs_big_meta = pd.read_csv(uscs_dir + f"meta_data/{config.dataset.lower()}_big_meta.csv", index_col='video_id', lineterminator='\n').reset_index()
    print(uscs_big_meta.shape)
    # print("Modify the input Subtitle CSV file  so that we have a subtitle dataframe")
    # subtitle_file = uscs_dir + f"transcripts/{config.dataset.lower()}_big_subtitle.csv"

    # uscs_big_subtitle = pd.read_csv(subtitle_file,  index_col='video_id', lineterminator='\n').reset_index()
    # assert 'video_id' in uscs_big_subtitle.columns, f"video_id has to be a column in {subtitle_file}"
    # assert 'subtitle' in uscs_big_subtitle.columns, f"subtitle has to be a column {subtitle_file}"

    # uscs_big_subtitle = uscs_big_subtitle[uscs_big_subtitle.video_id.isin(uscs_big_meta.video_id.unique())]
    # assert len(uscs_big_subtitle.video_id.unique()) == len(uscs_big_meta.video_id.unique()) , "Meta data and subtitle is NOT align in the number training..."

    # ### Appending the subtitle splitted
    # subtile_set = []
    
    # pbar = tqdm(total=len(uscs_big_meta.video_id.unique()))
    # for video_id in uscs_big_meta.video_id.unique():
    #     subtile = ' '.join(uscs_big_subtitle[uscs_big_subtitle.video_id==video_id].subtitle)
    #     subtile_set.append(subtile)
    #     pbar.update(1)
    # pbar.close()

    # uscs_big_subtitle = pd.DataFrame({"video_id":uscs_big_meta.video_id.unique(),
    #                              "subtitle": subtile_set,
    #                              "label": uscs_big_meta.onehot_label})
    # # Tokenize the subtitle
    # uscs_big_subtitle["Tokens"] = uscs_big_subtitle.apply(lambda uscs_big_subtitle :tokenizer_twitter_morphs(uscs_big_subtitle['subtitle']), axis=1)
    # # Save the files
    # uscs_big_subtitle.to_csv(uscs_dir + f"transcripts/{config.dataset.lower()}_big_subtitle.csv")
    uscs_big_subtitle = pd.read_csv(uscs_dir + f"transcripts/{config.dataset.lower()}_big_subtitle.csv", index_col='video_id', lineterminator='\n').reset_index()
    print(uscs_big_subtitle.shape)
    # define hyper-parameters
    num_features = 500
    min_word_count = 50 
    num_workers = 64    
    context = 5  
    downsampling = 1e-3  

    ### Generate KFOLD embedding
    with open(uscs_dir + "meta_data/k_fold_channel.json", 'rb') as fp:
            print("Load K FOLD by CHANNEL")
            k_fold_channel = pickle.load(fp)
            
    word2vec_dict = dict()
    tic = time()
    for i in range(5):
        print(f"FOLD: {i+1}")
        word2vec_dict[i] = dict()
        train_id, test_id = k_fold_channel[i]['train'], k_fold_channel[i]['test']
        print(np.max(train_id))
        x_train_all = uscs_big_subtitle.loc[train_id, "Tokens"]
        x_test_all = uscs_big_subtitle.loc[test_id, "Tokens"]
        
        model = word2vec.Word2Vec(x_train_all,
                            workers=num_workers,
                            vector_size=num_features,
                            min_count=min_word_count,
                            window=context,
                            sample=downsampling,
                            sg=1 )
        
        train_data_vecs = get_dataset(x_train_all, model, num_features)
        test_data_vecs =  get_dataset(x_test_all, model, num_features)
        word2vec_dict[i]['train'] = train_data_vecs
        word2vec_dict[i]['test'] = test_data_vecs
        
        word2vec_dict[i]['train'] = np.nan_to_num(word2vec_dict[i]['train'])
        word2vec_dict[i]['test'] =  np.nan_to_num(word2vec_dict[i]['test'])
        eta =  str(timedelta(seconds=int((time()-tic) / (i+1) * (5-i-1)))) 
        print('============== ETA: {} =============='.format(eta))
        del model
    
    with open(uscs_dir + "transcripts/word2vec_embedding_5fold.json", 'wb') as fp:
        pickle.dump(word2vec_dict, fp)
    
if __name__ == "__main__":
    with open('./config/config_pre.yaml', 'r') as stream:
        config = yaml.safe_load(stream)
    config = edict(config)
    
    main(config)