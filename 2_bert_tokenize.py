# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 11:53:56 2021

@author: jasmi
"""

import numpy as np
import pandas as pd
import torch
import transformers as ppb # pytorch transformers
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

'''Import Data'''
name = 'sample_10'
df = pd.read_csv('C:/Users/jasmi/Desktop/ML data/'+name+'.csv') 
df.head()

index = df['Unnamed: 0']
notes = df['text']
classfication = df['class']
notesCleaned = df['notesCleaned']
df.loc[df['class'] == 'suicide', 'classification_int'] = 1
df.loc[df['class'] == 'non-suicide', 'classification_int'] = 0


'''Importing pre-trained BERT model'''
model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')

'''Loading pretrained model/tokenizer'''
tokenizer = tokenizer_class.from_pretrained(pretrained_weights, padding=True, truncation=True, return_tensors="pt")
#model = model_class.from_pretrained(pretrained_weights)

'''Output'''
tokenized = notesCleaned.apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
tokens = pd.Series( [], dtype=str)

max_value = 0

print("\n",(df['notesCleaned']).shape, tokenized.shape)
i = 0
while i < len(df['notesCleaned']):
    #tokens[i] = np.pad(tokenized[i], (0, 512-len(tokenized[i])), 'constant', constant_values=(0))
    tokens[i] = tokenized[i]
    if len(tokenized[i]) >= max_value:
        max_value = len(tokenized[i])
    print(classfication[i], tokenized.shape, len(tokenized[i]), max_value)
    i = i + 1

'''Export'''
df['max_count'] = max_value
df['tokens'] = tokens
df.to_csv(name+'_tokens.csv', index=False)
