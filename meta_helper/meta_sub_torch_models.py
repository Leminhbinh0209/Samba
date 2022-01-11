import os
import sys
import numpy as np
import random
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import random
import math
import pickle
from .torch_utils import init_all

def _create_padding_mask(seq: torch.Tensor, pad_idx: int) -> torch.Tensor:
    """

    tensor([[[ True,  True,  True,  True, False, False, False]],
            [[ True,  True, False, False, False, False, False]],
            [[ True,  True,  True,  True,  True,  True, False]]])
    """

    return (seq == pad_idx)

def pad_collate(batch):
    lens_x = [len(i['sub_emb']) for i in batch]
    max_len = max(lens_x)
    #print(max_len)
    
    for i in batch:
        pad_mask = np.zeros(len(i['sub_emb']))
        assert len(i['sub_emb']) > 0
        if len(i['sub_emb']) < max_len:
            pad = np.zeros((max_len - len(i['sub_emb']), 768), dtype=np.float32)
            pad_mask = np.concatenate((pad_mask, np.ones(max_len - len(i['sub_emb']))))
            i['sub_emb'] = np.concatenate((i['sub_emb'], pad), 0)
        i['sub_emb'] = torch.tensor(i['sub_emb'])
        i['mask'] = torch.tensor(pad_mask).long()
    
    thumb_batch = torch.stack([torch.tensor(i['thumbnails']) for i in batch])
    head_batch = torch.stack([torch.tensor(i['headlines']) for i in batch])
    stat_batch = torch.stack([torch.tensor(i['statistics']) for i in batch])
    tag_batch = torch.stack([torch.tensor(i['video_tags']) for i in batch])
    sub_batch = torch.stack([torch.tensor(i['sub_emb']) for i in batch])
    lb_batch = torch.stack([torch.tensor(i['label']) for i in batch])
    mask_batch = _create_padding_mask(torch.stack([i['mask'] for i in batch]),1)
    return {"thumbnails":thumb_batch, 
            "headlines":head_batch,
            "statistics":stat_batch,  
            "video_tags":tag_batch, 
            "sub_emb":sub_batch, 
            "label":lb_batch, 
            "mask":mask_batch}
# Dataloader object

class MetaLoader(Dataset):
    def __init__(self, 
                 data_path, 
                 thumbnails, 
                 headlines, 
                 statistics, 
                 video_tags,
                 video_id,
                labels,
                mode='train') -> None:
        """
        Define our dataste loader object
        """
        self.data_path = data_path
        self.thumbnails = thumbnails
        self.headlines =  headlines
        self.statistics = statistics
        self.video_tags = video_tags
        self.video_id = video_id
        self.labels = labels
        self.mode = mode
        assert self.mode in ['train', 'test'], f"Unknown {mode}"
        assert len(self.labels) == len(self.headlines), "Data lengths are not equal ..."
        
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        vid = self.video_id[idx]
        dir_ = f"{self.data_path}/{self.mode}_moco/{vid}.bin"
        try:
            _, sub_emb, _ = pickle.load(open(dir_, 'rb'))
        except:
            print(dir_, "\n")
        assert sub_emb.shape[1] == 768, f"ERROR: lenght of subtitle is not 768"        
        label = self.labels[idx]
        data = {"thumbnails": self.thumbnails[idx],
               "headlines": self.headlines[idx],
                "statistics": self.statistics[idx],
                "video_tags": self.video_tags[idx],
                "sub_emb": sub_emb,
                "label": label,
                "idx": idx
               }
        return data
class ThumbnailNet(torch.nn.Module):
    def __init__(self, 
                    input_size=2048,
                    output_size = 128):
        """
        Thumbnail network
        """
        super().__init__()
        self.linear1 = torch.nn.Linear(input_size, output_size) if output_size is not None else torch.nn.Identity()
        self.output_size = output_size if output_size is not None else input_size
        self.dropout = nn.Dropout(p=0.25)
        
    def forward(self, x):
        """
        x: B x d
        """
        
        x_out = self.linear1(self.dropout(x))
        return x_out

class HeadlineNet(torch.nn.Module):
    def __init__(self, 
                 headlines_vocab_size,
                 headlines_words_seq_length,
                 embedding_vector_length=32,
                 text_dropout = 0,
                 output_size = 128
                 ):
        """
        Headline network
        """
        super().__init__()
        self.headlines_vocab_size = headlines_vocab_size
        self.headlines_words_seq_length = headlines_words_seq_length
        self.embedding_vector_length = embedding_vector_length
        self.text_dropout = text_dropout
        self.output_size = output_size
        
        self.embedding = torch.nn.Embedding(self.headlines_vocab_size+1, 
                                            self.embedding_vector_length, 
                                            max_norm=True)
        
        self.lstm = torch.nn.LSTM(input_size=self.embedding_vector_length, 
                                  hidden_size=self.output_size, 
                                  num_layers=1,
                                  batch_first=True,
                                  dropout=self.text_dropout)
    def init_hidden(self, batch_size):
        return(autograd.Variable(torch.randn(1, batch_size, self.output_size)),
                        autograd.Variable(torch.randn(1, batch_size, self.output_size)))
    
    def forward(self, x):

        """
        x: B x L
        """
        x_embed = self.embedding(x) # B x L x embedding_vector_length
        outputs, (ht, ct) = self.lstm(x_embed)
        output = ht[-1]
        return output 

class StaticticsNet(torch.nn.Module):
    def __init__(self, 
                    input_size=25,
                    output_size = 128):
        """
        Statictics network
        """
        super().__init__()
        self.output_size = output_size
        self.linear1 = torch.nn.Linear(input_size, output_size)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        """
        """
        x_out = self.relu(self.linear1(x))
        return x_out
class VideoTagsNet(torch.nn.Module):
    def __init__(self, 
                 video_tags_vocab_size,
                 video_tags_seq_length,
                 embedding_vector_length=32,
                 text_dropout = 0,
                 output_size = 128
                 ):
        """
        Video Tags network
        """
        super().__init__()
        self.video_tags_vocab_size = video_tags_vocab_size
        self.video_tags_seq_length = video_tags_seq_length
        self.embedding_vector_length = embedding_vector_length
        self.text_dropout = text_dropout
        self.output_size = output_size
        
        self.embedding = torch.nn.Embedding(self.video_tags_vocab_size+1, 
                                            self.embedding_vector_length, 
                                            max_norm=True)
        
        self.lstm = torch.nn.LSTM(input_size=self.embedding_vector_length, 
                                  hidden_size=self.output_size, 
                                  num_layers=1,
                                  batch_first=True,
                                  dropout=self.text_dropout)
    def forward(self, x):

        """
        x: B x L
        """
        x_embed = self.embedding(x) # B x L x embedding_vector_length
        output, (hidden, cell) = self.lstm(x_embed)
        output = hidden[-1] 
        return output

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    
class TransformerClassifier(nn.Module):
    """
    Pretrained Bert model to predict output
    """
    def __init__(self, device = 'cuda:0', strategy='Mean',output_size = 128):
        super(TransformerClassifier, self).__init__()

        self.device = device
        self.strategy = strategy
        self.d_model = 768  
        self.output_size = output_size
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=4, dropout=0.5)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1) #nn.LayerNorm(self.d_model)
        self.drop = nn.Dropout(0.5)
        # self.PositionalEncoding = PositionalEncoding(self.d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model), requires_grad=True)
        self.fc0 = nn.Linear(self.d_model, self.output_size)


    def forward(self, features,mask=None):

        # Preprocess
       # print(features.shape)
        # seq x batch x dim
        cls_tokens = self.cls_token.expand(features.size(0), 1, self.d_model)
        cls_mask = (torch.zeros(features.size(0), 1) == 1).to(self.device)
        #print(mask.shape, cls_mask.shape, features.shape)
        mask = torch.cat((cls_mask, mask), 1)
        features = torch.cat((cls_tokens, features), 1)
        features = features.permute(1, 0, 2)#torch.relu(self.fc_emb(features)).permute(1, 0, 2)
        # features = self.PositionalEncoding(features)
       # print(mask.shape)
        features = self.transformer_encoder(features, src_key_padding_mask=mask)
        # batch x seq x dim
        features = features.permute(1, 0, 2)
        #print(features.shape)
        if self.strategy == 'Mean':

            # batch x dim
            mask = ((mask == False) * 1.).unsqueeze(-1)
            features = features * mask
            #print(features.sum(1).shape, mask.sum(1).shape)
            assert (mask.sum(1) > 0).all()
            features = features.sum(1) / mask.sum(1)
        else:
            features = features[:,0, ...]
        #print(features.shape)
        features = torch.tanh(self.fc0(features))

        return features
# REF: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention

class MultiheadAttention(nn.Module):
    
    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3*embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o
class MetaverseNet(torch.nn.Module):
    def __init__(self, 
                 thumbnail_size,
                 statistics_size,
                 headlines_vocab_size,
                 headlines_words_seq_length,
                 video_tags_vocab_size,
                 video_tags_seq_length,
                 embedding_vector_length=32,
                 text_dropout = 0,
                 output_size = 128,
                 apply_attention = False,
                 apply_pooling = False,
                 drop_out = 0.5,
                 device='cuda'
                 ):
        """
        Metaverse ()>__<)  network
        """
        super(MetaverseNet, self).__init__()
        self.device = device
        self.apply_attention = apply_attention
        self.apply_pooling = apply_pooling
        self.output_size = output_size
        self.drop_out = drop_out
        
        self.thumbnail_net = ThumbnailNet(input_size=thumbnail_size, output_size=self.output_size)
        
        self.statistics_net = StaticticsNet(input_size=statistics_size, output_size=self.output_size)
        
        self.headline_net = HeadlineNet(headlines_vocab_size,
                                         headlines_words_seq_length,
                                         embedding_vector_length,
                                         text_dropout,
                                         self.output_size)
        
        self.videotag_net = VideoTagsNet(video_tags_vocab_size,
                                         video_tags_seq_length,
                                         embedding_vector_length,
                                         text_dropout,
                                         self.output_size)
        
        self.subtitle_net = TransformerClassifier(device=self.device,  strategy='pool', output_size=self.output_size)
        

        self.dropout = nn.Dropout(p=self.drop_out)
        if self.apply_attention:
            self.fc0 = nn.Linear(self.output_size, 256)
            self.fc1 = nn.Linear(256, 1) 
        
        elif self.apply_pooling:
            self.rnn = nn.GRU(input_size=self.output_size, 
                            hidden_size=self.output_size //2, 
                            num_layers=2, 
                            batch_first=True,  
                            dropout=0.5)
            self.out_layer = torch.nn.Linear(self.output_size //2, 2)

        else:
            self.out_layer = torch.nn.Linear(self.thumbnail_net.output_size+\
                                             self.statistics_net.output_size+\
                                             self.headline_net.output_size+\
                                             self.videotag_net.output_size+\
                                             self.subtitle_net.output_size, 2) # 5 model

    def get_params(self):
        thumbnail_params = list(self.thumbnail_net.parameters())
        stats_params = list(self.statistics_net.parameters())
        headline_params = list(self.headline_net.parameters())
        video_params = list(self.videotag_net.parameters())
        subtitle_params = list(self.subtitle_net.parameters())
        general_params =  list(self.out_layer.parameters())  # + list(self.rnn.parameters()) 
        if self.apply_attention:
            print("Add attention param ")
            general_params += list(self.fc0.parameters()) + list(self.fc1.parameters())
        elif self.apply_pooling:
            print("Add pooling param ")
            general_params += list(self.rnn.parameters()) 
        
        return thumbnail_params, stats_params, headline_params, video_params, subtitle_params, general_params
        
    def forward(self, x, mode='train'):

        """
        x: B x L
        """
        thumbnail_emb = self.thumbnail_net(x["thumbnails"]) # B x output_size
        statistics_emb = self.statistics_net(x["statistics"]) # B x output_size
        headline_emb = self.headline_net(x["headlines"].to(torch.int32)) # B x output_size
        videotag_emb = self.videotag_net(x["video_tags"].to(torch.int32)) # B x output_size
        subtitle_emb = self.subtitle_net (x["sub_emb"], x["mask"]) # B x output_size 
        x_embs = torch.cat(( thumbnail_emb, statistics_emb, headline_emb,  videotag_emb, subtitle_emb), dim=1)
        # Apply gated linear output
       # Apply non-local attention
        if self.apply_attention:
            features = torch.cat((thumbnail_emb, statistics_emb, headline_emb,  videotag_emb, subtitle_emb), dim=1).reshape(-1, 5, self.output_size)
            #print(features.shape)
            features_1 = self.dropout(torch.tanh(self.fc0(features)))
            # N x S x 1
            features_1 = self.fc1(features_1).squeeze(-1)
        
            # N x S x 1
            features_1 = torch.softmax(features_1, 1).unsqueeze(2)
            #print(features_1.shape)
            # N x S x D
            x_embs = torch.einsum('nsk,nsd->nkd', (features_1,features )).squeeze(1)

        elif self.apply_pooling:
            features = x_embs.reshape(-1, 5, self.output_size)
            _, x_embs = self.rnn(features)
            x_embs = x_embs[-1]

        x_embs = self.dropout(x_embs)
        out = self.out_layer(x_embs) # B x  2
        return out 