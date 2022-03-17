# Samba: Identifying Inappropriate Videos for Young Children on YouTube
## Overview of our framework
<p align="center">
    <img src="https://i.ibb.co/Dbppg5T/main-architecture.png" width="960" alt="overall pipeline">
<p>
    
## 1. Installation
- Ubuntu 18.04.5 LTS
- CUDA 11.3
- Python 3.8.12


## 2. Dataset

Download our subttitle and meta data, training and test data from our homepage. Then organize the dataset folder as follows:

```
./data
  |--YOUTUBE_data
          |---meta_data
          |---transcripts


```

## 3. Data pre-processing


#### 3.1 Download thumbnail photos

Download and embed the thumbnail images using InceptionNet, or you can download embedding vector directly from our homepage.

```
$ python thumbnail_download.py
```

#### 3.2 Meta feature embedding

Embed all the features: headline, tags, style, thumbnail by running:

```
$ python meta_embedding.py
```

Embedding data is exported to `meta_embedding.hdf5` that have structure:

```
  {video_id: 1-D list,
  thumbnail: 2-D matrix,
  headline: 2-D matrix,
  style: 2-D matrix,
  tags:2-D matrix,
  y: 1-D array}
```

#### 3.3 Word2vec subtitle embedding

_Note: This process may take time since the pre-trained word2vec model used to embedding subtitle._

Using video index in the `*big_meta.csv` file and and put subtitle in that order. Change the input csv file name at **line 54**. This file has to have `video_id` and `subtitle` columns. Then, run:

```
$ python subtitle_pre.py
```

## 4. Training

#### 4.1 Meta training

Modify the ML method in `config/config_meta.yaml` file at **line 3**. Start training by runining:

```
$ python meta_train.py
```

#### 4.2 Word2Vec training

Modify the ML method in `config/config_word2vec.yaml` file at **line 3**. Start training by runining:

```
$ python subtitle_word2vec_train.py
```

#### 4.3 SAMBA model
Follow the README document in ```pretrain``` folder to create subtitles' embeddings. The run:


```
$ python train.py
```
Our pre-trained model can be obtained [here](https://drive.google.com/file/d/1TreS_HP6lERjJiTZKQkAhYuXI6s_FT1f/view?usp=sharing)
