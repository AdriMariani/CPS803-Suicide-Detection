# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 10:23:02 2021

@author: jasmi
"""

import pandas as pd
import re


''' Import'''
df = pd.read_csv('Suicide_Detection.csv') 
#print(df.shape)
#print(df.head())

''' Functions'''
def a(text):
    chars = "\*_{}[]()#+-!$';<>|:%=¸”&‚"
    for c in chars:
        text = text.replace(c, " ")
    text = text.replace('"', " ")
    text = text.replace(".", " ")
    text = text.replace(",", " ")
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")
    text = text.replace("'", " ")
    text = text.replace("filler", "'")
    text = text.replace("  ", " ")
    text = text.lower()
    text = text.encode("ascii", "ignore")
    text = text.decode()
    return text

'''Processing'''
notes = df['text']
classfication = df['class']
notesCleaned = pd.Series( [], dtype=str)

for i in range(0, len(notes)):
    each = (a(notes[i]))
    if i%10000 == 0:
        print("At record = %s" %(i))
    notesCleaned[i] = each

df['notesCleaned'] = notesCleaned

'''Export'''
df.to_csv('c_Suicide_Detection.csv', index=False)
