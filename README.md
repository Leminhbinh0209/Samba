# Youtube Video Detection

## 1. Dataset

We use `*big_meta.csv` file as a standard and split data. The dataset folder is organized as follow:

```
./data
  |--USCS_data
  |        |---meta_data
  |        |---transcripts
  |
  |
  |--CURVE_data
          |---meta_data
          |---transcripts


```

- For the USCS dataset, download the meta data csv file [here](https://drive.google.com/file/d/1RtN7aIjP7JMI4GA4HaB6UaDmulLyGBPR/view?usp=sharing) and put into the `meta_data` folder, and the subtitle input file [here](https://drive.google.com/file/d/17C0mZoLzL8hslV-V2pfP5QxwHPe1c4Af/view?usp=sharing) and put to the `transcripts` folder.
- For the CURVE dataset, download the meta data csv file [here](https://drive.google.com/drive/u/1/folders/1A-6fbxJTOOTBnUmgxQ3qhMiDj4rysLsu), and the subtitle [here](https://drive.google.com/drive/u/1/folders/19FmlSkpgy30t_3jAxpTk5vzgM9z0d9IG).

## 2. Data pre-processing

#### 2.1 Train-test split

**Dev Stage**: Please use pre-splited data can be downloaded directly [here](https://drive.google.com/drive/u/1/folders/1DxemCy87tvfS2C_NUdWV-OA3zAwGd948) and [here](https://drive.google.com/drive/u/1/folders/1pRgR6hMH2Z_ccOPuWDBNc0eyu05Je5mP).

Modify dataset name (i.e., USCS or CURVE) in config file `./config/config_pre.yaml`, then run:

```
$ python train_test_split.py
```

#### 2.2 Download thumbnail photos

Download and embed the thumbnail images using InceptionNet

```
$ python thumbnail_download.py
```

#### 2.3 Meta feature embedding

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

#### 2.4 Word2vec subtitle embedding

_Note: This process may take time since the pre-trained word2vec model used to embedding subtitle._

Using video index in the `*big_meta.csv` file and and put subtitle in that order. Change the input csv file name at **line 54**. This file has to have `video_id` and `subtitle` columns. Then, run:

```
$ python subtitle_pre.py
```

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

Change ensemble methods at **line 30** of `average_ensemble.py` and run

```
$ python average_ensemble.py
```
