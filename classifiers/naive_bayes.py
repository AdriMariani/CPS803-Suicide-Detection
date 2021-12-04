import pandas as pd
import numpy as np
import utils
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn import metrics

'''Import Training Data'''
train_name = 'Suicide_Detection_tokens1'
df = pd.read_csv(train_name+'.csv') 
train_df = df.loc[df['limit'] == 0] #filter records where tokens <512
#train_df.head()

'''Import Test Data'''
test_name = 'reddit_depression_suicidewatch_tokens1'
test_df = pd.read_csv(test_name+'.csv') 
#train_df.head()

'''Training Dataframes'''
index = train_df['id']
notes = train_df['text']
classfication = train_df['class']
notesCleaned = train_df['notesCleaned']
classfication_int = train_df['classification_int']
tokens = train_df['tokens']
max_count = train_df['max_count']

test_index = test_df['id']
test_notes = test_df['text']
test_classfication = test_df['class']
test_notesCleaned = test_df['notesCleaned']
test_classfication_int = test_df['classification_int']
test_tokens = test_df['tokens']
test_max_count = test_df['max_count']

''' Additional Columns '''
#finalpredictions = pd.Series( [], dtype=str)

'''Constants'''
if max_count[0] > 512 or test_max_count[0]:
    padding = 512
else:
    padding = max(max_count[0], test_max_count[0])

train_matrix = utils.create_matrix(tokens, padding)
test_matrix = utils.create_matrix(test_tokens, padding)

'''Bernoulli Naive Bayes Classification'''
bnb = BernoulliNB(alpha=1.0, binarize=0.0, fit_prior=True, class_prior=None)
bnb = bnb.fit(train_matrix, classfication_int)
predictionsNB = bnb.predict(test_matrix)
scoreNB = metrics.accuracy_score(test_classfication_int, predictionsNB)
#print(predictionsNB)
#test_df['predictionsNB'] = predictionsNB
print("Bernoulli Naive Bayes Prediction Score:", round(scoreNB*100, 2), "%")

'''Gaussian Naive Bayes Classification'''
gnb = GaussianNB(var_smoothing=1e-5)
gnb = gnb.fit(train_matrix, classfication_int)
predictionsGNB = gnb.predict(test_matrix)
scoreGNB = metrics.accuracy_score(test_classfication_int, predictionsGNB)
#print(predictionsGNB)
#test_df['predictionsGNB'] = predictionsGNB
print("Gaussian Naive Bayes Prediction Score:", round(scoreGNB*100, 2), "%")