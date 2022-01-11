#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 14:48:38 2021

@author: anonymous
"""
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BertForSequenceClassification
import torch
import math

class BertClassifier(nn.Module):
    """
    Pretrained Bert model to predict output
    """
    def __init__(self, bert_model = 'bert-base-uncased',device = 'cuda:0', freeze_bert = False, strategy='Mean'):
        super(BertClassifier, self).__init__()

        print('BertModel')
        self.bert_layer = BertModel.from_pretrained(
            bert_model, # Use the 12-layer BERT model, with an uncased vocab.
            add_pooling_layer=False,
            
        )
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)
        self.device = device
        self.strategy = strategy

        #self.bert_layer = self.bert_layer.to(self.device)
        self.d_model = 768
        self.fc0 = nn.Linear(self.d_model, 256)
        self.fc1 = nn.Linear(256, 1)        
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False

    def bertify_input(self, sentences):
        '''
        Preprocess the input sentences using bert tokenizer and converts them to a torch tensor containing token ids
        
        Args:
            sentences (list): source sentences
        Returns:
            token_ids (tensor): tokenized sentences | size: [BS x S]
            attn_masks (tensor): masks padded indices | size: [BS x S]
            input_lengths (list): lengths of sentences | size: [BS]
        '''
        #print(len(sentences))
        # Tokenize all of the sentences and map the tokens to thier word IDs.
        input_ids = []
        attention_masks = []

        # For every sentence...
        for sent in sentences:
            # `encode_plus` will:
            #   (1) Tokenize the sentence.
            #   (2) Prepend the `[CLS]` token to the start.
            #   (3) Append the `[SEP]` token to the end.
            #   (4) Map tokens to their IDs.
            #   (5) Pad or truncate the sentence to `max_length`
            #   (6) Create attention masks for [PAD] tokens.
            encoded_dict = self.bert_tokenizer.encode_plus(
                                sent,                      # Sentence to encode.
                                add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                                max_length = 144,           # Pad & truncate all sentences.
                                pad_to_max_length = True,
                                return_attention_mask = True,   # Construct attn. masks.
                                return_tensors = 'pt',     # Return pytorch tensors.
                           )
            
            # Add the encoded sentence to the list.    
            input_ids.append(encoded_dict['input_ids'])
            
            # And its attention mask (simply differentiates padding from non-padding).
            attention_masks.append(encoded_dict['attention_mask'])
        
        # Convert the lists into tensors.
        input_ids = torch.cat(input_ids, dim=0).cuda(self.device, non_blocking=True)#.to(self.device)
        attention_masks = torch.cat(attention_masks, dim=0).cuda(self.device, non_blocking=True)#.to(self.device)



        return input_ids, attention_masks

    def forward(self,  input):
        '''
        Feed the batch of sentences to a BERT encoder to obtain contextualized representations of each token
        
        Args:
            sentences (list): source sentences
        Returns:
            cont_reps (tensor): BERT Embeddings | size: [BS x S x d_model]
            token_ids (tensor): tokenized sentences | size: [BS x S]
        '''
        if isinstance(input, tuple):
            token_ids, attn_masks = input
        else:
            # Preprocess sentences
            token_ids, attn_masks = self.bertify_input(input)


        out_dict = self.bert_layer(token_ids, attention_mask = attn_masks)

        if self.strategy == 'Mean':
            features = out_dict['last_hidden_state']#[0]
            mask = (attn_masks == 0)#.unsqueeze(-1)
                    # N x S x 256
            features_1 = torch.tanh(self.fc0(features))
            # N x S x 1
            features_1 = self.fc1(features_1)
            # N x S
            #print(features_1.shape, mask.shape)
            features_1 = features_1.squeeze(2).masked_fill(
                mask,
                float("-inf"),
            )
            # N x S x 1
            features_1 = torch.softmax(features_1, 1).unsqueeze(2)
    
            # N x S x D
            to_pool = torch.einsum('nsk,nsd->nkd', (features_1,features )).reshape(-1, self.d_model)

        else:
            to_pool = out_dict['pooler_output']

        out = to_pool
        #print(out.shape)
        return out
    
