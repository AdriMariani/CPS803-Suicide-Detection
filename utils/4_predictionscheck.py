import pandas as pd
import numpy as np
import utils 
from transformers import BertTokenizer
bert = BertTokenizer.from_pretrained("bert-base-cased")
from tokenizers import BertWordPieceTokenizer
#tok = BertWordPieceTokenizer('./data/bert-vocabs/bert-base-cased-vocab.txt', lowercase=False)


'''Import Training Data'''
name = 'reddit_depression_suicidewatch_tokens1_predictions'
df = pd.read_csv(name+'.csv') 
#df = df.loc[df['classification_int'] == 1] 
#df = df.loc[df['predictionsLR'] == 1] 

'''Dataframes'''
index = df['id']
notes = df['text']
classfication = df['class']
notesCleaned = df['notesCleaned']
classfication_int = df['classification_int']
tokens = df['tokens']
max_count = df['max_count']
prediction = df['predictionsLR'] 

tf_score = {}
idf_score = {}
tf_score1 = {}
idf_score1 = {}
tf_score2 = {}
idf_score2 = {}
for i in range(0, len(index)):
    # true positives
    if classfication_int[i] == prediction[i] and classfication_int[i]==1:
        sentence = notesCleaned[i]
        tf_score_dict_truepos = utils.word_tokenize(sentence, tf_score)
        idf_score_dict_truepos = utils.word_tokenize(sentence, idf_score)
    # false positives
    elif prediction[i] == 1 and classfication_int[i] == 0:
        sentence = notesCleaned[i]
        tf_score_dict_falsepos = utils.word_tokenize(sentence, tf_score1)
        idf_score_dict_falsepos = utils.word_tokenize(sentence, idf_score1)
    # false negatives
    elif prediction[i] == 0 and classfication_int[i] == 1:
        sentence = notesCleaned[i]
        tf_score_dict_falseneg = utils.word_tokenize(sentence, tf_score2)
        idf_score_dict_falseneg = utils.word_tokenize(sentence, idf_score2)


n = 25
print(name) 
tf_idf_score_truepositive = {key: tf_score_dict_truepos[key] * idf_score_dict_truepos.get(key, 0) for key in tf_score.keys()}
lst = utils.get_top_n(tf_idf_score_truepositive, n)
print("True Positives")
print(lst)

tf_idf_score_falsepositive = {key: tf_score_dict_falsepos[key] * idf_score_dict_falsepos.get(key, 0) for key in tf_score1.keys()}
lst = utils.get_top_n(tf_idf_score_falsepositive, n)
print("\nFalse Positives")
print(lst)

tf_idf_score_falsenegative = {key: tf_score_dict_falseneg[key] * idf_score_dict_falseneg.get(key, 0) for key in tf_score2.keys()}
lst = utils.get_top_n(tf_idf_score_falsenegative, n)
print("\nFalse Negatives")
print(lst)