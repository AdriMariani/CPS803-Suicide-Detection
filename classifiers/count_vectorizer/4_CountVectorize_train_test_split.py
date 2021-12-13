'''
- Using CountVectorize
- Splitting training and testing datasets
- Dataset: Suicide_Detection
- Input folder: tokenized_datasets
- Output folder: predicted_datasets
'''

import warnings
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
warnings.filterwarnings("ignore")

'''Import Dataset '''
name = 'Suicide_Detection'
df = pd.read_csv("tokenized_datasets/"+name+'.csv') 
df = df.loc[df['limit'] == 0] #filter records where tokens <512

''' Splitting Method '''
total_size=len(df)
train_size=int(np.floor(0.75*total_size))
train=df.head(train_size)
test=df.tail(len(df) - train_size)
#train.to_csv('tokenized_datasets/'+'train.csv')
#test.to_csv('tokenized_datasets/'+test.csv')

'''Import Training Data'''
train_name = 'train'
train_df = pd.read_csv('tokenized_datasets/'+train_name+'.csv', header=0)
#print(df.iloc[0])

train_df['notesCleaned'].replace('', np.nan, inplace=True)
train_df.dropna(subset=['notesCleaned'], inplace=True)
sentences_train = train_df['notesCleaned'].values
train_ex = sentences_train.shape[0]
y_train = np.zeros((train_ex))
y_train[train_df['class'] == 'suicide'] = 1

'''Import Test Data'''
test_name = 'test'
test_df = pd.read_csv('tokenized_datasets/'+test_name+'.csv') 

test_df['notesCleaned'].replace('', np.nan, inplace=True)
test_df.dropna(subset=['notesCleaned'], inplace=True)
sentences_test = test_df['notesCleaned'].values
test_ex = sentences_test.shape[0]
y_test = np.zeros((test_ex))
y_test[test_df['class'] == 'suicide'] = 1
test_df['classification_int'] = y_test

'''Vectorize & tokenize'''
vectorizer = CountVectorizer()
vectorizer.fit(sentences_train)
X_train = vectorizer.transform(sentences_train)
X_test  = vectorizer.transform(sentences_test)

'''LR Classification'''
from sklearn.linear_model import LogisticRegression
classifierLR = LogisticRegression().fit(X_train, y_train)
score = classifierLR.score(X_test, y_test)
print('Accuracy for Logistic Regression model: {:.4f}'.format(score))

''' Bernoulli Naive Bayes '''
from sklearn.naive_bayes import BernoulliNB
classifierBNB = BernoulliNB().fit(X_train, y_train)
score = classifierBNB.score(X_test, y_test)
print('Accuracy for Bernoulli NB model: {:.4f}'.format(score))

''' Multinomial Naive Bayes '''
from sklearn.naive_bayes import MultinomialNB
classifierMNB = MultinomialNB()
classifierMNB = classifierMNB.fit(X_train, y_train)
predictionsMNB = classifierMNB.predict(X_test)
scoreLR = metrics.accuracy_score(y_test, predictionsMNB)
print('Accuracy for Multinomial NB model: {:.4f}'.format(scoreLR))

''' SVM '''
from sklearn.svm import LinearSVC
classifierSVC = LinearSVC()
classifierSVC = classifierSVC.fit(X_train, y_train)
predictionsSVC = classifierSVC.predict(X_test)
scoreSVC = metrics.accuracy_score(y_test, predictionsSVC)
print('Accuracy for Linear SVC model: {:.4f}'.format(scoreSVC))

'''MLP'''
from keras.models import Sequential
from keras import layers
from keras.backend import clear_session
clear_session()

input_dim = X_train.shape[1]  # Number of features
model = Sequential()
model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
#model.summary()

history = model.fit(X_train, y_train, epochs=10, verbose=False, validation_data=(X_test, y_test), batch_size=100)
#loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
#print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Accuracy for MLP model:  {:.4f}".format(accuracy))
predictionsMLP = model.predict(X_test)
test_df['predictions_'] = predictionsMLP     #MLP
test_df.loc[test_df['predictions_'] >= 0.5, 'predictions'] = 1
test_df.loc[test_df['predictions_'] < 0.5, 'predictions'] = 0
predictionsMLP = test_df['predictions']

confusion_matrix = metrics.confusion_matrix(y_test, predictionsMLP)

'''Export predictions as CSV'''
#test_df['predictions'] = predictLR         #Logistic Regression
#test_df['predictions'] = predictionsSVM    #SVM
#test_df['predictions'] = predictionsNB     #Naive Bayes
#test_df['predictions'] = predictionsMNB    #Multinomial Naive Bayes
#test_df.to_csv('predicted_datasets/'+test_name+'_CountVectorize.csv', index=False)

test_df.to_csv('predicted_datasets/'+test_name+'_MLP_CountVectorize.csv', index=False)
