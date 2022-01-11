## 1. Pretraining

#### 1.1 MoCo pre-training

First, your sequences should be pre-processed and saved as .csv file:

```
$ python moco_pretrain.py --data [DATA.csv] --moco-k 512 --moco-t 0.07 --moco-dim 768 --batch-size 128
```

#### 1.2 MoCo Document embedding

First, create a folder where embeddings will be saved:

```
$ python embedd_docs.py --document [DATA.csv] --save [SAVE_FOLDER] --checkpoint [BERT ENCODER CHECKPOINT]
```
