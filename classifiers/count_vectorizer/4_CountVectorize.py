'''
- Using CountVectorize
- Splitting training and testing datasets
- Training Dataset: Suicide_Detection
- Test Datasets: reddit_depression_suicidewatch, suicide_notes
- Input Datasets: cleaned_datasets
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
import pickle

'''Import Training Data'''
train_name = 'Suicide_Detection'
train_df = pd.read_csv('cleaned_datasets/'+train_name+'.csv', header=0)
#print(df.iloc[0])

train_df['notesCleaned'].replace('', np.nan, inplace=True)
train_df.dropna(subset=['notesCleaned'], inplace=True)
sentences_train = train_df['notesCleaned'].values
train_ex = sentences_train.shape[0]
y_train = np.zeros((train_ex))
y_train[train_df['class'] == 'suicide'] = 1

'''Import Test Data'''
#test_name = 'suicide_notes' 
test_name = 'reddit_depression_suicidewatch'    
test_df = pd.read_csv("cleaned_datasets/"+test_name+'.csv') 

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

print(test_name)
'''LR Classification'''
from sklearn.linear_model import LogisticRegression
classifierLR = LogisticRegression(random_state=0, tol=1e-5, max_iter=100000, dual=False)
classifierLR = classifierLR.fit(X_train, y_train)
predictLR = classifierLR.predict(X_test)
scoreLR = metrics.accuracy_score(y_test, predictLR)
print('Accuracy for Logistic Regression model: {:.4f}'.format(scoreLR))

''' Bernoulli Naive Bayes '''
from sklearn.naive_bayes import BernoulliNB
classifierBNB = BernoulliNB().fit(X_train, y_train)
predictionsBNB = classifierBNB.predict(X_test)
scoreBNB = metrics.accuracy_score(y_test, predictionsBNB)
print('Accuracy for Bernoulli NB model: {:.4f}'.format(scoreBNB))

''' Multinomial Naive Bayes '''
from sklearn.naive_bayes import MultinomialNB
classifierMNB = MultinomialNB()
classifierMNB = classifierMNB.fit(X_train, y_train)
predictionsMNB = classifierMNB.predict(X_test)
scoreMNB = metrics.accuracy_score(y_test, predictionsMNB)
print('Accuracy for Multinomial NB model: {:.4f}'.format(scoreMNB))

''' SVM '''
from sklearn.svm import LinearSVC
classifierSVC = LinearSVC()
classifierSVC = classifierSVC.fit(X_train, y_train)
predictionsSVM = classifierSVC.predict(X_test)
scoreSVM = metrics.accuracy_score(y_test, predictionsSVM)
print('Accuracy for Linear SVC model: {:.4f}'.format(scoreSVM))

'''MLP Classification'''
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
loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
#print("Training Accuracy: {:.4f}".format(accuracy))
#loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Accuracy for MLP model:  {:.4f}".format(accuracy))
predictionsMLP = model.predict(X_test)
test_df['predictions_'] = predictionsMLP     #MLP
test_df.loc[test_df['predictions_'] >= 0.5, 'predictions'] = 1
test_df.loc[test_df['predictions_'] < 0.5, 'predictions'] = 0
predictionsMLP = test_df['predictions']

print("Last record: ",predictionsMLP[-1])
confusion_matrix = metrics.confusion_matrix(y_test, predictionsMLP)
#tn, fp, fn, tp = lr_confusion_matrix.ravel()
print("Confusion Matrix:", confusion_matrix)
print("Precision:", precision_score(y_test, predictionsMLP))
from sklearn.metrics import recall_score
print("Recall Score:",recall_score(y_test, predictionsMLP))
print("F1 score:", f1_score(y_test, predictionsMLP))
#print("Sensitivity/Positive Recall:", calc_sensitivity(lr_confusion_matrix))
#print("Specificity/Negative Recall:", calc_specificity(lr_confusion_matrix))
#print("------------")

'''Export predictions as CSV'''
#test_df['predictions'] = predictLR         #Logistic Regression
#test_df['predictions'] = predictionsSVM    #SVM
#test_df['predictions'] = predictionsNB     #Naive Bayes
#test_df['predictions'] = predictionsMNB    #Multinomial Naive Bayes
#test_df['predictions'] = predictionsMLP    #MLP
#test_df.to_csv('predicted_datasets/'+test_name+'_CountVectorize.csv', index=False)
#test_df.to_csv('predicted_datasets/'+test_name+'_MLP_CountVectorize.csv', index=False)
