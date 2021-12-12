'''
- Separating true positives, false positives and false negatives
- Test Datasets: test, reddit_depression_suicidewatch, suicide_notes
- Input Datasets: predicted_datasets
- Output folder: words
'''
import pandas as pd
import numpy as np
import utils 

'''Import Training Data'''

name = 'test_CountVectorize'
#name = 'test_BERT'
#name = 'suicide_notes_BERT'
#name = 'suicide_notes_CountVectorize'
#name = 'reddit_depression_suicidewatch_BERT'
#name = 'reddit_depression_suicidewatch_CountVectorize'
df = pd.read_csv('predicted_datasets/'+name+'.csv') 

'''Dataframes'''
index = df['id']
notes = df['text']
classfication = df['class']
notesCleaned = df['notesCleaned']
classfication_int = df['classification_int']
#tokens = df['tokens']
#max_count = df['max_count']
prediction = df['predictions'] 

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

n = 100
print(name) 
tf_idf_score_truepositive = {key: tf_score_dict_truepos[key] * idf_score_dict_truepos.get(key, 0) for key in tf_score.keys()}
lst = utils.get_top_n(tf_idf_score_truepositive, n)
#print("True Positives")
with open('words/true_positives_'+name+'.csv', 'w') as f:
    for key in lst:
        val = lst[key]*10000
        val = np.ceil(val)
        #rint(val)
        if val < 1:
            val = 1
        val = str(int(val))
        f.write(key+"\t"+val+"\n")

tf_idf_score_falsepositive = {key: tf_score_dict_falsepos[key] * idf_score_dict_falsepos.get(key, 0) for key in tf_score1.keys()}
lst = utils.get_top_n(tf_idf_score_falsepositive, n)
with open('words/false_positives_'+name+'.csv', 'w') as f:
    for key in lst:
        val = lst[key]*10000
        val = np.ceil(val)
        if val < 1:
            val = 1
        val = str(int(val))
        f.write(key+"\t"+val+"\n")
#print("\nFalse Positives")
#print(lst)

tf_idf_score_falsenegative = {key: tf_score_dict_falseneg[key] * idf_score_dict_falseneg.get(key, 0) for key in tf_score2.keys()}
lst = utils.get_top_n(tf_idf_score_falsenegative, n)
with open('words/false_negatives_'+name+'.csv', 'w') as f:
    for key in lst:
        val = lst[key]*10000
        val = np.ceil(val)
        if val < 1:
            val = 1
        val = str(int(val))
        f.write(key+"\t"+val+"\n")
#print("\nFalse Negatives")
#print(lst)