import numpy as np
import pandas as pd
import torch
import transformers as ppb # pytorch transformers
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
from tokenizers import BertWordPieceTokenizer

'''Import Data'''
name = 'c_Suicide_Detection'
df = pd.read_csv('C:/Users/jasmi/Desktop/ML data/'+name+'.csv') 
#print(df.head())

index = df['id']
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
tokens = pd.Series( [], dtype=str)
token_count = pd.Series( [], dtype=str)
limit = pd.Series( [], dtype=str) #more than 512?
max_value = 0

#print("\n",(df['notesCleaned']).shape, tokenized.shape)
i = 0
while i < len(df['notesCleaned']):
    str_ = str(notesCleaned[i])
    tokenized = tokenizer.encode(str_, add_special_tokens=True)
    #print(str_)
    #print(tokenizer.decode(tokenized))
    val_len = len(tokenized)
    #print(i, index[i])
    token_count[i] = val_len
    try:
        tokens[i] = tokenized
        if val_len >= max_value:
            max_value = val_len
    except KeyError:
        tokens[i] = ""
        max_value = 0
    except ValueError:
        print(notesCleaned[i])
    
    if val_len > 512:
        limit[i] = 1
        tokens[i]  = ""
    else:
        limit[i] = 0
    #print(classfication[i], len(tokenized), max_value)
    i = i + 1

'''Export'''
df['max_count'] = max_value
df['tokens'] = tokens
df['token_count'] = token_count
df['limit'] = limit
df.to_csv(name+'_tokens.csv', index=False)