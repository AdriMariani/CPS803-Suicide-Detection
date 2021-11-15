# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 10:23:02 2021

@author: jasmi
"""

import pandas as pd
import re

'''Files'''
train_path='Suicide_Detection.csv', 
train_path2='suicide_notes.csv',
test_path='reddit_depression_suicidewatch.csv'

'''Input'''
path = train_path     ## Change this filename

''' Functions'''
def convertTuple(tup):
    # initialize an empty string
    str = ''
    for item in tup:
        str = str + item
    return str

def a(text):
    chars = "\/*_{}[]()#+-!$';<>|:%=¸”&‚"
    text = text.replace('"', " ")
    text = text.replace(".", " ")
    text = text.replace("\n", "")
    text = text.replace("\t", " ")
    for c in chars:
        text = text.replace(c, " ")
    text = text.replace("'", " ")
    text = text.replace("filler", "'")
    text = text.replace("  ", " ")
    text = text.lower()
    text = text.encode("ascii", "ignore")
    text = text.decode()
    return text

def split(text):
    s = text.split(" ")
    return len(s)

''' Import'''
name = convertTuple(path)
name = name.replace('.csv','')
df = pd.read_csv(name+'.csv') 
#print(df.shape)
    
'''Processing'''
notes = df['text']
df = df.drop('text', 1)
if name == 'Suicide_Detection':
    classfication = df['class']  
elif name == 'suicide_notes':
    df['class'] = 'suicide'
    classfication = 'suicide' 
elif name == 'reddit_depression_suicidewatch':
    df.loc[df['label'] == 'SuicideWatch', 'class'] = 'suicide'
    df.loc[df['label'] == 'depression', 'class'] = 'non-suicide'
    classfication = df['class']  

notesCleaned = pd.Series( [], dtype=str)

for i in range(0, len(notes)):
    each = (a(notes[i]))
    if i%10000 == 0:
        print("At record = %s" %(i))
    notesCleaned[i] = each

df['notesCleaned'] = notesCleaned

'''Export'''
df.to_csv('c_'+name+'.csv', index=False)
