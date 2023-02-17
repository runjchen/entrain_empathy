#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
from collections import defaultdict
import pickle

import matplotlib.pyplot as plt
from matplotlib.pyplot import errorbar, boxplot
import seaborn as sns

from scipy import stats
import random

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix

from tqdm import tqdm

import nltk
from nltk.tokenize import word_tokenize
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertTokenizer, BertModel

import time
import datetime

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
print('Device count:', cuda.device_count())
print('Using device:', device)

#Additional Info when using cuda
if device == 'cuda':
  print(torch.cuda.get_device_name(0))
  print(torch.cuda.get_device_properties(0))
  print('Memory Usage:')
  print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
  print('Cached: ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')
  
# # Load Entrain Encoder
checkpoint = torch.load('checkpoint_5.pt')
entrainer = BertModel.from_pretrained('bert-base-uncased')
entrainer.load_state_dict(checkpoint['bert_state_dict'])


# # Load EmpatheticDialogues

# Load Training Data - EmpatheticDialogues

# !wget https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/empatheticdialogues.tar.gz
# !tar -xf empatheticdialogues.tar.gz

TRAIN_PATH = 'empatheticdialogues/train.csv'
VAL_PATH = 'empatheticdialogues/valid.csv'
TEST_PATH = 'empatheticdialogues/test.csv'

def load_data(path):
  '''dataframe from data path'''
  file = open(path).readlines()
  f = [line.strip().split(",")[:8] for line in file] # remove the retrieval data 
  df = pd.DataFrame(f[1:], columns=f[0])

  df['prompt'] = df['prompt'].str.replace("_comma_", ",")
  df['utterance'] = df['utterance'].str.replace("_comma_", ",")

  return df


def get_context_sent(df):
  contexts = []
  sentences = []
  for did, df_i in df.groupby('conv_id'):
    
    prompt = df_i['prompt'].values[0]
    history = [prompt]
    for sent in df_i['utterance'].tolist():
      contexts.append(' '.join(history))
      sentences.append(sent)

      # update context
      history.append(sent)

  return pd.DataFrame(zip(contexts, sentences), columns=['contexts', 'sentences'])


train_data = get_context_sent(load_data(TRAIN_PATH))
val_data = get_context_sent(load_data(VAL_PATH))
test_data = get_context_sent(load_data(TEST_PATH))
print(len(train_data), len(val_data), len(test_data))

from nlp import Dataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

batch_size = 16
train_dataset = Dataset.from_pandas(train_data)
train_dataloader = DataLoader(
            train_dataset,
            sampler = RandomSampler(train_dataset),
            batch_size = batch_size
        )


val_dataset = Dataset.from_pandas(val_data)
val_dataloader = DataLoader(
            val_dataset,
            sampler = SequentialSampler(val_dataset),
            batch_size = batch_size
        )

test_dataset = Dataset.from_pandas(test_data)
test_dataloader = DataLoader(
            test_dataset,
            sampler = RandomSampler(test_dataset),
            batch_size = batch_size
        )


## Model

class BERT_encoder(nn.Module):
    def __init__(self, config, bert, entrainer):
        super(BERT_encoder, self).__init__()
        
        self.config = config
        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert_tokenizer.bos_token = self.bert_tokenizer.cls_token # CLS token will work as BOS token
        self.bert_tokenizer.eos_token = self.bert_tokenizer.sep_token # SEP token will work as EOS token
        
        self.bert = bert
        self.fc1 = nn.Linear(768+768,768)
        self.fc2 = nn.Linear(768, 768)
        
        #activation functions
        self.dropout1 = nn.Dropout(0.1)
        self.relu1 =  nn.ReLU()
        self.relu2 =  nn.ReLU()
        
        # entrainer 
        self.entrainer = entrainer
        for param in self.entrainer.parameters():
            param.requires_grad = False
        
    #define the forward pass
    def forward(self, sent):
        sent_id, mask = self.bert_tokenizer(sent)
        #pass the inputs to the model  
        bert_out = self.bert(sent_id, attention_mask=mask).pooler_output #(batch_size, hidden_size)
        entrain_out = self.entrainer(sent_id, attention_mask=mask).pooler_output
        
        x = self.fc1(torch.cat((bert_out, entrain_out), dim=1))
        x = self.relu1(x)                   
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu2(x)             
    
        return x

encoder = BERT_encoder(BertConfig.from_pretrained('bert-base-uncased'),
                       BertModel.from_pretrained('bert-base-uncased'),
                      entrainer)


# using architecture from https://www.guru99.com/seq2seq-model.html
# https://jasmijn.ninja/annotated_encoder_decoder/

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, MAX_LENGTH=512):
        super().__init__()
        
        #initialize the encoder and decoder
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, source, target, teacher_forcing_ratio=0.5):
        input_length = max([len(s) for s in source]) #get the input length (number of words in sentence)
        batch_size = len(target)
        target_length = max([len(s) for s in target])
#         vocab_size = self.decoder.output_dim
        vocab_size = 50257
        #initialize a variable to hold the predicted outputs
        outputs = torch.zeros(target_length, batch_size, vocab_size).to(self.device)

        #encode every word in a sentence
        for i in range(input_length):
            encoder_output, encoder_hidden = self.encoder(source[i])

        #use the encoder’s hidden layer as the decoder hidden
        decoder_hidden = encoder_hidden.to(device)
        
        #add a token before the first predicted word
        decoder_input = torch.tensor([SOS_token], device=device)  # SOS

        #topk is used to get the top K value over a list
        #predict the output word from the current target word. If we enable the teaching force,  then the #next decoder input is the next word, else, use the decoder output highest value. 

        for t in range(target_length):   
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs[t] = decoder_output
            teacher_force = random.random() < teacher_forcing_ratio
            topv, topi = decoder_output.topk(1)
            input = (target[t] if teacher_force else topi)
            if(teacher_force == False and input.item() == EOS_token):
                break

        return outputs
      
      
model = Seq2Seq(encoder=encoder, 
                decoder=GPT2LMHeadModel.from_pretrained("gpt2"),
               device=device)


## Training

# This training code is based on the `run_glue.py` script here:
# https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

# Set the seed value all over the place to make this reproducible.

seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

training_stats = []

from transformers import AdamW
criterion = nn.NLLLoss()
optimizer = AdamW(model.parameters(), lr = 2e-5, eps = 1e-8)
from transformers import get_linear_schedule_with_warmup
epochs = 5
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0,
                                            num_training_steps = total_steps)

total_t0 = time.time() # Measure the total training time for the whole run.

for epoch_i in range(0, epochs):
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    
    ## =================== Training =================== ##
    t0 = time.time()
    total_train_loss = 0
    
    model.train() # turn on dropout layers
    loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
    
    for step, batch in loop:
        b_contexts = batch['contexts']
        b_labels = batch['sentences']
        
        optimizer.zero_grad() 

        ### forward pass ###
        b_train_pred = model(b_contexts, b_labels)
        loss = criterion(b_train_pred, b_labels)
        total_train_loss += loss.item() * b_input_ids.shape[0]
        
        ### backward pass ###
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) #avoid gradient explode
        optimizer.step()
        scheduler.step()

        ### update progress bar ###
        loop.set_postfix(loss=loss.item())
    
    avg_train_loss = total_train_loss / train_dataloader.dataset.__len__()        
    training_time = format_time(time.time() - t0)
    
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(training_time)) 
        
    ## =================== Validation =================== ##
    t0 = time.time()

    model.eval()

    # Tracking variables 
    total_eval_loss = 0

    # Evaluate data for one epoch
    for batch in tqdm(val_dataloader):
        
        b_contexts = batch['contexts']
        b_labels = batch['sentences']
        
        with torch.no_grad():        
            b_train_pred = model(b_contexts, b_labels)
            loss = criterion(b_train_pred, b_labels)
            
        total_eval_loss += loss.item()*b_input_ids.shape[0]
    
    avg_val_loss = total_eval_loss / validation_dataloader.dataset.__len__()
    validation_time = format_time(time.time() - t0)

    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))
    
    # Record all statistics from this epoch.
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Training Time': training_time,
            'Validation Time': validation_time,
        }
    )
    
    #save model
    torch.save({
            'epoch': epoch_i+1,
            'model_state_dict': model.state_dict(),
            'bert_state_dict':model.bert.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict':scheduler.state_dict(),
            'avg_train_loss': avg_train_loss,
            }, f'./model_checkpoint_{epoch_i+1}.pt')

print("Training complete!")

print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))



## Evaluation 

from datasets import load_metric
bleu_metric = load_metric('bleu')

preds = []
labels = []
for batch in tqdm(test_dataloader):
    b_contexts = batch['contexts']
    b_labels = batch['sentences']

    with torch.no_grad():        
        b_train_pred = model(b_contexts, b_labels)
        preds.extend(b_train_pred)
        labels.extend(b_labels)

score = bleu_metric.compute(predictions=preds, 
                              references=[[s] for s in labels],)

print(score)
