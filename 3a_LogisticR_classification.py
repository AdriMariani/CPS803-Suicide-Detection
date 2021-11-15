# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 15:07:03 2021

@author: jasmi
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
import utils

'''Import Training Data'''
train_name = 'sample_20'
train_df = pd.read_csv(train_name+'_tokens.csv') 
#train_df.head()

'''Import Test Data'''
test_name = 'sample_10'
test_df = pd.read_csv(test_name+'_tokens.csv') 
#train_df.head()

'''Training Dataframes'''
index = train_df['Unnamed: 0']
notes = train_df['text']
classfication = train_df['class']
notesCleaned = train_df['notesCleaned']
classfication_int = train_df['classification_int']
tokens = train_df['tokens']
max_count = train_df['max_count']

test_index = test_df['Unnamed: 0']
test_notes = test_df['text']
test_classfication = test_df['class']
test_notesCleaned = test_df['notesCleaned']
test_classfication_int = test_df['classification_int']
test_tokens = test_df['tokens']
test_max_count = test_df['max_count']

'''Constants'''
padding = max(max_count[0], test_max_count[0])

'''Functions'''
train_matrix = utils.create_matrix(tokens, padding)
test_matrix = utils.create_matrix(test_tokens, padding)

'''Classification'''
clf = LogisticRegression(max_iter=1200000)
predict_ = clf.fit(train_matrix, classfication_int)
predict_ = clf.predict(test_matrix)
score = clf.score(test_matrix, test_classfication_int)
print("Prediction Score:", round(score*100, 2), "%")
