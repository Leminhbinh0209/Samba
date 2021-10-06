# Youtube Kid-Video Detection

## 1. Dataset

We use `*big_meta.csv` file as a standard and split data. The dataset folder is organized as follow:

```
./data
  |--USCS_data
  |        |---meta_data
  |        |---transcript
  |        |---word2vec
  |
  |--CURVE_data
          |---meta_data
          |---transcript
          |---word2vec

```

## 2. Pre-process data

#### 2.1 Generate K-Fold split

Modify dataset name (i.e., USCS or CURVE) in config file `./config/config_pre.yaml`, then run:

```
$ python kfold_split.py
```

#### 2.2 Download thumbnail photos

Download thumbnail and embed the thumbnail using InceptionNet

```
$ python thumbnail_download.py
```

#### 2.3 Embeb meta features

Embed all the features: headline, tags, style, thumbnail by running:

```
$ python meta_embedding.py
```

#### 2.4 Create subtitle CVS file

_Note: This process may take time since the pre-trained word2vec model used to embedding subtitle._

Using video index in the `*big_meta.csv` file and and put subtitle in that order. Change the input csv file name at **line 54**. This file has to have `video_id` and `subtitle` columns. Then, run:

```
$ python subtitle_pre.py
```

After running, `*big_subtitle.csv` will be created. Morover, the average word2vec embedding is save in 5 fold split at `transcripts/word2vec_embedding_5fold.json`.

## 3. Training

#### 3.1 Meta training

Modify the ML method in `config/config_meta.yaml` file at **line 3**. Start training by runining:

```
$ python meta_train.py
```

#### 3.2 Word2Vec training

Modify the ML method in `config/config_word2vec.yaml` file at **line 3**. Start training by runining:

```
$ python subtitle_word2vec_train.py
```

#### 3.3 Average ensemble

Change ensemble methods at **line 30**

```
$ python average_ensemble.py
```