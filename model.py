#!/usr/bin/env python
# coding: utf-8

# In[14]:


import os
import numpy as np
import pandas as pd
from collections import defaultdict
import pickle

# import parselmouth
# from parselmouth.praat import call

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


# In[16]:


get_ipython().system('nvidia-smi')


# # Load Entrain Encoder

# In[17]:


# !unzip "Copy of checkpoint_5.zip"


# In[18]:


class BERT_Arch(nn.Module):
  def __init__(self, bert, output_size=12):
    super(BERT_Arch, self).__init__()
    self.bert = bert
    # self.fc = nn.Linear(768, output_size)
    self.fc1 = nn.Linear(768,128)
    self.fc2 = nn.Linear(128, output_size)
    
    #activation functions
    self.dropout1 = nn.Dropout(0.1)
    # self.dropout2 = nn.Dropout(0.1)
    self.relu1 = nn.ReLU()
    # self.relu2 = nn.ReLU()
    # self.softmax = nn.LogSoftmax(dim=1)

  def forward(self, sent_id, mask):
    bert_out = self.bert(sent_id, attention_mask=mask).pooler_output #(batch_size, hidden_size)
    x = self.fc1(bert_out)
    x = self.relu1(x)
    x = self.dropout1(x)

    x = self.fc2(x)
    # x = self.relu2(x)
    # x = self.dropout2(x)

    # x = self.fc3(x)
    # x = self.softmax(x)
    return x


# In[21]:


checkpoint = torch.load('checkpoint_5.pt')
bert = BertModel.from_pretrained('bert-base-uncased')
bert.load_state_dict(checkpoint['bert_state_dict'])
model = BERT_Arch(bert)
model.load_state_dict(checkpoint['model_state_dict'])
model.output_hidden_states=True
model.eval()


# # Load EmpatheticDialogues

# In[ ]:





# In[ ]:


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


# In[ ]:


with torch.no_grad():
    for batch in tqdm(dataloader):
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        b_outputs = model(b_input_ids,mask=b_input_mask)
        hidden_states = model.encoder_hidden_states #(batch_size, sequence_length, hidden_size)
        hidden_states = hidden_states.mean(dim=1) #(batch_size, hidden_size)


# # Model

# In[13]:





# In[ ]:




