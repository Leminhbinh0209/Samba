# Samba: Identifying Inappropriate Videos for Young Children on YouTube<br>

<img alt="GitHub top language" src="https://img.shields.io/github/languages/top/Leminhbinh0209/Samba?style=for-the-badge" height="25"  onmouseover="this.height='60'" onmouseout="this.height='25'" ><img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/Leminhbinh0209/Samba?style=for-the-badge" height="25"><img alt="GitHub repo size" src="https://img.shields.io/github/repo-size/Leminhbinh0209/Samba?style=for-the-badge" height="25">
<br />
_31st ACM International Conference on Information & Knowledge Management, Georgia, USA, 2022_
## Overview of our framework
<p align="center">
    <img src="https://i.ibb.co/RbXFFKj/main-architecture-v3.png" width="900" alt="overall pipeline">
<p>
<!--     <img src="https://i.ibb.co/RbXFFKj/main-architecture-v3.png" alt="main-architecture-v3" border="0"> -->
    
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
#
*Star if you find it useful.* ⭐
