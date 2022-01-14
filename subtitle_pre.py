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
    youtube_dir = f"{config.data_folder}/{config.dataset}_data/" 
    youtube_big_meta = pd.read_csv(youtube_dir + f"meta_data/{config.dataset.lower()}_big_meta.csv", index_col='video_id', lineterminator='\n').reset_index()
    print(youtube_big_meta.shape)
    print("Modify the input Subtitle CSV file  so that we have a subtitle dataframe")
    subtitle_file = youtube_dir + f"transcripts/{config.dataset.lower()}_big_subtitle.csv"

    youtube_big_subtitle = pd.read_csv(subtitle_file,  index_col='video_id', lineterminator='\n').reset_index()
    assert 'video_id' in youtube_big_subtitle.columns, f"video_id has to be a column in {subtitle_file}"
    assert 'subtitle' in youtube_big_subtitle.columns, f"subtitle has to be a column {subtitle_file}"

    youtube_big_subtitle = youtube_big_subtitle[youtube_big_subtitle.video_id.isin(youtube_big_meta.video_id.unique())]
    assert len(youtube_big_subtitle.video_id.unique()) == len(youtube_big_meta.video_id.unique()) , "Meta data and subtitle is NOT align in the number training..."

    ### Appending the subtitle splitted
    subtile_set = []
    
    pbar = tqdm(total=len(youtube_big_meta.video_id.unique()))
    for video_id in youtube_big_meta.video_id.unique():
        subtile = ' '.join(youtube_big_subtitle[youtube_big_subtitle.video_id==video_id].subtitle)
        subtile_set.append(subtile)
        pbar.update(1)
    pbar.close()

    youtube_big_subtitle = pd.DataFrame({"video_id":youtube_big_meta.video_id.unique(),
                                 "subtitle": subtile_set,
                                 "label": youtube_big_meta.onehot_label})
    # Tokenize the subtitle
    youtube_big_subtitle["Tokens"] = youtube_big_subtitle.apply(lambda youtube_big_subtitle :tokenizer_twitter_morphs(youtube_big_subtitle['subtitle']), axis=1)
    # Save the files
    youtube_big_subtitle.to_csv(youtube_dir + f"transcripts/{config.dataset.lower()}_big_subtitle.csv")
    youtube_big_subtitle = pd.read_csv(youtube_dir + f"transcripts/{config.dataset.lower()}_big_subtitle.csv", index_col='video_id', lineterminator='\n').reset_index()
    print(youtube_big_subtitle.shape)
    # define hyper-parameters
    num_features = 500
    min_word_count = 50 
    num_workers = 64    
    context = 5  
    downsampling = 1e-3  

    f = open(f"{youtube_dir}{config.dataset.lower()}_train_videos.txt", "r")
    train_video_id = f.readlines()
    train_video_id = [i.strip() for i in train_video_id]
    f.close()

    f = open(f"{youtube_dir}{config.dataset.lower()}_test_videos.txt", "r")
    test_video_id = f.readlines()
    test_video_id = [i.strip() for i in test_video_id]
    f.close()

    index_lookup = dict(zip(youtube_big_subtitle.video_id.values, np.arange(len(youtube_big_subtitle))))
    train_id = [index_lookup[u] for u in train_video_id if u in index_lookup]
    test_id = [index_lookup[u] for u in test_video_id if u in index_lookup]
    word2vec_dict = dict()
    tic = time()
    print(np.max(train_id))
            
    word2vec_dict = dict()
    tic = time()
  
   
    x_train_all = youtube_big_subtitle.loc[train_id, "Tokens"]
    x_test_all = youtube_big_subtitle.loc[test_id, "Tokens"]
    
    model = word2vec.Word2Vec(x_train_all,
                        workers=num_workers,
                        vector_size=num_features,
                        min_count=min_word_count,
                        window=context,
                        sample=downsampling,
                        sg=1 )
    
    train_data_vecs = get_dataset(x_train_all, model, num_features)
    test_data_vecs =  get_dataset(x_test_all, model, num_features)
    word2vec_dict['train'] = train_data_vecs
    word2vec_dict['test'] = test_data_vecs
    
    word2vec_dict['train'] = np.nan_to_num(word2vec_dict[i]['train'])
    word2vec_dict['test'] =  np.nan_to_num(word2vec_dict[i]['test'])
    eta =  str(timedelta(seconds=int((time()-tic) / (i+1) * (5-i-1)))) 
    print('============== ETA: {} =============='.format(eta))
    del model
    
    with open(youtube_dir + "transcripts/word2vec_embedding_5fold.json", 'wb') as fp:
        pickle.dump(word2vec_dict, fp)
    
if __name__ == "__main__":
    with open('./config/config_pre.yaml', 'r') as stream:
        config = yaml.safe_load(stream)
    config = edict(config)
    
    main(config)