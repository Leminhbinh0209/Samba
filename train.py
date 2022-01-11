import warnings
warnings.filterwarnings("ignore")
import os
import sys
import numpy as np
import torch
from  torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import  accuracy_score
from sklearn.metrics import  precision_score, recall_score, f1_score
import json
from easydict import EasyDict as edict
import yaml
from transformers import  AdamW
from meta_helper.h5py_func import read_dict
from meta_helper.meta_sub_torch_models import *
from meta_helper.torch_loss import *
from meta_helper.torch_utils import *


with open('./config/config_ensemble.yaml', 'r') as stream:
        config = yaml.safe_load(stream)
config = edict(config)
os.environ ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU
print(f"Runing with GPUs: {config.GPU}")
seed_everything(config.random_seed)

   
def main():
    uscs_dir = f"{config.data_folder}/data/{config.dataset}_data/" 
    data_dir = f"{uscs_dir}/meta_data/meta_emb/"

    with open(data_dir + 'voc.json') as outfile:
            voc = json.load( outfile)
    headlines_vocab_size, headlines_words_seq_length = voc['headline']['voc_size'], voc['headline']['max']
    video_tags_vocab_size, video_tags_seq_length = voc['tag']['voc_size'], voc['tag']['max']

    print("Read embedding data...")
    instance_keys = ['thumbnail', 'headline', 'style', 'tags', 'y']
    inputs = read_dict(data_dir +  'meta_embedding.hdf5')

    all_video_id = [u.decode('UTF-8') for u in inputs['video_id'].tolist()]
    all_thumbnails_features = np.asarray(inputs[instance_keys[0]], dtype=np.float32)       
    all_headlines_features =  np.asarray(inputs[instance_keys[1]], dtype=np.float32)    
    all_statistics_features = np.asarray(inputs[instance_keys[2]], dtype=np.float32)    
    all_video_tags_features = np.asarray(inputs[instance_keys[3]], dtype=np.float32)    
    targets = inputs['y']

    assert all_headlines_features.shape[1] == headlines_words_seq_length, "Headline leng issue!!!"
    assert all_video_tags_features.shape[1] == video_tags_seq_length, "Tags length issue!!!"
    print("Number of samples : ", len(targets) ) 
    print("Number positive samples: ", len(targets) - np.sum(targets) )

    f = open(f"{uscs_dir}{config.dataset.lower()}_train_videos.txt", "r")
    train_video_id = f.readlines()
    train_video_id = [i.strip() for i in train_video_id]
    f.close()

    f = open(f"{uscs_dir}{config.dataset.lower()}_test_videos.txt", "r")
    test_video_id = f.readlines()
    test_video_id = [i.strip() for i in test_video_id]
    f.close()

    index_lookup = dict(zip(all_video_id, np.arange(len(all_video_id))))
    train_val_set_indices = [index_lookup[u] for u in train_video_id if u in index_lookup]
    test_set_indices = [index_lookup[u] for u in test_video_id if u in index_lookup]

    thumbnails_num_examples = len(all_thumbnails_features)
    dataset_labels = targets
    dataset_labels_binary = targets
    dataset_pred_binary = np.zeros_like(dataset_labels_binary)

    # TRAIN_VAL_video_ids = np.take(dataset_videos, train_set_indices, axis=0)
    Y_train_val = np.take(dataset_labels, indices=train_val_set_indices, axis=0)
    Y_train_val_binary = np.take(dataset_labels_binary, indices=train_val_set_indices, axis=0)

    # TEST_videos_ids = np.take(dataset_videos, test_set_indices, axis=0)
    Y_test = np.take(dataset_labels, indices=test_set_indices, axis=0)
    Y_test_binary = np.take(dataset_labels_binary, indices=test_set_indices, axis=0)

    # Split TRAIN to TRAIN & VAL (basically get the indices)
    indices_train, indices_val = stratified_train_val_split(set_labels=Y_train_val_binary, val_size=config.validation_split)

    # Get Y_train and Y_val
    Y_train = np.take(Y_train_val, indices=indices_train, axis=0)
    Y_train_binary = np.take(Y_train_val_binary, indices=indices_train, axis=0)
    Y_val = np.take(Y_train_val, indices=indices_val, axis=0)
    Y_val_binary = np.take(Y_train_val_binary, indices=indices_val, axis=0)

    # Get video ID of train val
    id_train = np.take(train_video_id, indices=indices_train, axis=0).tolist()
    id_val = np.take(train_video_id, indices=indices_val, axis=0).tolist()

    # Split TRAIN to TRAIN & VAL (basically get the indices)
    indices_train, indices_val = stratified_train_val_split(set_labels=Y_train_val_binary, val_size=config.validation_split)

    # Get Y_train and Y_val
    Y_train = np.take(Y_train_val, indices=indices_train, axis=0)
    Y_train_binary = np.take(Y_train_val_binary, indices=indices_train, axis=0)
    Y_val = np.take(Y_train_val, indices=indices_val, axis=0)
    Y_val_binary = np.take(Y_train_val_binary, indices=indices_val, axis=0)

    # Get video ID of train val
    id_train = np.take(train_video_id, indices=indices_train, axis=0).tolist()
    id_val = np.take(train_video_id, indices=indices_val, axis=0).tolist()

    """
    THUMBNAILS
    """
    # TRAIN & VAL
    X_train_val_thumbnails = np.take(all_thumbnails_features, indices=train_val_set_indices, axis=0)
    X_train_thumbnails = np.take(X_train_val_thumbnails, indices=indices_train, axis=0)
    X_val_thumbnails = np.take(X_train_val_thumbnails, indices=indices_val, axis=0)
    X_test_thumbnails = np.take(all_thumbnails_features, indices=test_set_indices, axis=0)
    """
    HEADLINES
    """
    # TRAIN & VAL
    X_train_val_headlines = np.take(all_headlines_features, train_val_set_indices, axis=0)
    X_train_headlines = np.take(X_train_val_headlines, indices=indices_train, axis=0)
    X_val_headlines = np.take(X_train_val_headlines, indices=indices_val, axis=0)
    X_test_headlines = np.take(all_headlines_features, indices=test_set_indices, axis=0)
    """
    STATISTICS
    """
    # TRAIN & VAL
    X_train_val_statistics = np.take(all_statistics_features, train_val_set_indices, axis=0)
    X_train_statistics = np.take(X_train_val_statistics, indices=indices_train, axis=0)
    X_val_statistics = np.take(X_train_val_statistics, indices=indices_val, axis=0)
    X_test_statistics = np.take(all_statistics_features, indices=test_set_indices, axis=0)
    """
    VIDEO TAGS
    """
    # TRAIN & VAL
    X_train_val_video_tags = np.take(all_video_tags_features, indices=train_val_set_indices, axis=0)
    X_train_video_tags = np.take(X_train_val_video_tags, indices=indices_train, axis=0)
    X_val_video_tags = np.take(X_train_val_video_tags, indices=indices_val, axis=0)
    X_test_video_tags = np.take(all_video_tags_features, indices=test_set_indices, axis=0)

    train_dataset = MetaLoader(data_path=f"{config.data_folder}/data/YOUTUBE_data/sub_embedding/",
                            thumbnails=X_train_thumbnails, 
                                headlines=X_train_headlines, 
                                statistics=X_train_statistics, 
                                video_tags=X_train_video_tags,
                                video_id=id_train,
                                labels=Y_train_binary,
                                mode='train')

    val_dataset = MetaLoader(data_path=f"{config.data_folder}/data/YOUTUBE_data/sub_embedding/",
                            thumbnails=X_val_thumbnails, 
                                headlines=X_val_headlines, 
                                statistics=X_val_statistics, 
                                video_tags=X_val_video_tags,
                                video_id=id_val,
                                labels=Y_val_binary, 
                                mode='train')

    test_dataset = MetaLoader(data_path=f"{config.data_folder}/data/YOUTUBE_data/sub_embedding/",
                            thumbnails=X_test_thumbnails, 
                                headlines=X_test_headlines, 
                                statistics=X_test_statistics, 
                                video_tags=X_test_video_tags,
                                video_id=test_video_id,
                                labels=Y_test_binary, 
                                mode='test')
    def _init_fn(worker_id):
        np.random.seed(int(config.random_seed))
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True, collate_fn=pad_collate, worker_init_fn=_init_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True, collate_fn=pad_collate)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True, collate_fn=pad_collate)

    metanet = MetaverseNet(thumbnail_size=2048,
                            statistics_size=25,
                            headlines_vocab_size=headlines_vocab_size,
                            headlines_words_seq_length=headlines_words_seq_length,
                            video_tags_vocab_size=video_tags_vocab_size,
                            video_tags_seq_length=video_tags_seq_length,
                            embedding_vector_length=32,
                            text_dropout = 0,
                            output_size = 256,
                            apply_attention = True,
                            apply_pooling=False,
                            drop_out=0.4,
                            device='cuda')

     
    metanet = metanet.cuda()
    print("Total training paprameters: ", numel(metanet, True))
    
    checkpoint_dir = f"./checkpoints/{config.running_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    thumbnail_params, stats_params, headline_params, video_params, subtitle_params, general_params = metanet.get_params()

    criterion = SmoothCrossEntropyLoss(smoothing=0.2)
    optimizer = optim.Adam([{'params': subtitle_params, 'lr': 2e-4},
                           {'params': thumbnail_params, 'lr': 1e-3},
                           {'params': stats_params, 'lr': 1e-3},
                           {'params': headline_params, 'lr': 1e-3},
                           {'params': video_params, 'lr': 1e-3},
                           {'params': general_params, 'lr': 1e-3}])  

    scaler = torch.cuda.amp.GradScaler()
    
    metanet.train()
    best_acc = 0.0
    patience = 0
    for epoch in range(config.NB_EPOCHS):
        metanet.train()
        train_loss = AverageMeter()
        train_acc = AverageMeter()
      
        for batch_idx, data in enumerate(train_dataloader):
          
            lb = data["label"]
            lb = lb.to(torch.long).cuda()
            for k in data:
                if k in ["headlines", "video_tags"]:
                    data[k] = data[k].to(torch.int32).cuda()
                if k in ["mask"]:
                    data[k] = data[k].cuda()
                else:
                    data[k] = data[k].to(torch.float32).cuda()
            #with torch.cuda.amp.autocast():
            out = metanet(data)
    
            loss = criterion(out, lb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc1 = accuracy(out, lb)
            train_loss.update(loss.item() , lb.size(0))
            train_acc.update(acc1[0], lb.size(0))
            
            sys.stdout.write("\r")
            sys.stdout.flush()
            sys.stdout.write("Train Epoch: {e:02d} Batch: {batch:04d}/{size:04d} |  Loss:{loss:.4f} | Acc: {acc:.4f}"\
                            .format(e=epoch+1, batch=batch_idx+1, size=len(train_dataloader),\
                                    loss=train_loss.avg, acc=train_acc.avg))
     
        metanet.eval()
        with torch.no_grad():
            val_loss = AverageMeter()
            val_acc = AverageMeter()
            for batch_idx, data in enumerate(val_dataloader):
                lb = data["label"]
                lb = lb.to(torch.long).cuda()
                for k in data:
                    if k in ["headlines", "video_tags"]:
                        data[k] = data[k].to(torch.int32).cuda()
                    if k in ["mask"]:
                        data[k] = data[k].cuda()
                    else:
                        data[k] = data[k].to(torch.float32).cuda()
                #with torch.cuda.amp.autocast():
                out = metanet(data)
                loss = criterion(out, lb)
                acc1 = accuracy(out, lb)
                val_loss.update(loss.item() , lb.size(0))
                val_acc.update(acc1[0], lb.size(0))
            
            print("\nValiadtion Loss: {loss:.4f} | Acc: {acc:.4f}"\
                .format(loss=val_loss.avg, acc=val_acc.avg))
            # best validation acc
            best = False
            if val_acc.avg > best_acc:
                print("Val Acc \033[0;32m improved \033[0;0m from {acc_past:.4f} to {acc_new:.4f} ".format(acc_past=best_acc, acc_new=val_acc.avg))
                best_acc = val_acc.avg
                best = True
                patience = 0
            else:
                patience += 1
            save_model(checkpoint_dir, epoch, metanet, optimizer,  best)
       
      

    pretrained_weights = torch.load(str('_'.join([checkpoint_dir, 'best.pth'])))['state_dict']
    metanet.load_state_dict(pretrained_weights)
    metanet.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        val_loss = AverageMeter()
        val_acc = AverageMeter()
        for batch_idx, data in enumerate(test_dataloader):
            lb = data["label"]
            lb = lb.to(torch.long).cuda()
            for k in data:
                if k in ["headlines", "video_tags"]:
                    data[k] = data[k].to(torch.int32).cuda()
                if k in ["mask"]:
                    data[k] = data[k].cuda()
                else:
                    data[k] = data[k].to(torch.float32).cuda()
            out = metanet(data)
            loss = criterion(out, lb)
            acc1 = accuracy(out, lb)
            val_loss.update(loss.item() , lb.size(0))
            val_acc.update(acc1[0], lb.size(0))
            
            y_true = np.concatenate((y_true, lb.cpu().detach().numpy()), axis=0) if len(y_true) else lb.cpu().detach().numpy()
            y_pred = np.concatenate((y_pred, out.cpu().detach().numpy()), axis=0) if len(y_pred) else out.cpu().detach().numpy()
        print("\nTest Loss:{loss:.4f} | Acc: {acc:.4f}"\
            .format(loss=val_loss.avg, acc=val_acc.avg))
    y_pred = np.argmax(y_pred, axis=1)
    AVERAGE_USED = 'macro'
    test_accuracy = accuracy_score(1-y_true, 1-y_pred)
    test_precision = precision_score(1-y_true, 1-y_pred, average=AVERAGE_USED)
    test_recall = recall_score(1-y_true, 1-y_pred, average=AVERAGE_USED)
    test_f1_score = f1_score(1-y_true,1-y_pred, average=AVERAGE_USED)

    print('\033[92mTEST Accuracy:\t %.3f' % (test_accuracy))
    print('TEST Precision:\t %.3f' % (test_precision))
    print('TEST Recall:\t %.3f' % (test_recall))
    print('TEST F1-Score:\t %.3f' % (test_f1_score))
    print('=== FINISH ===\033[0m')

if __name__ == '__main__':
    main()