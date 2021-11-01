# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 11:53:56 2021

@author: jasmi
"""

import numpy as np
import pandas as pd
import torch
import transformers as ppb # pytorch transformers

#tokenizer.encode("a visually stunning rumination on love", add_special_tokens=True)

#Import Data
train_df = pd.read_csv('C:/ML data/bert_test_data.csv') 
train_df.head()

index = train_df['Unnamed: 0']
notes = train_df['text']
classfication = train_df['class']
notesCleaned = train_df['notesCleaned']

#Importing pre-trained BERT model
model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')

#Loading pretrained model/tokenizer
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)

#Output
tokenized = train_df['notesCleaned'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
