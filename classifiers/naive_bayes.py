import pandas as pd
import numpy as np
import sys
sys.path.append('..')
import utils.utils as utils
import utils.evaluation as evaluation
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

path = '../datasets/c_Suicide_Detection.csv'
df = pd.read_csv(path, header=0)

sentences = df['notesCleaned'].values

ex = sentences.shape[0]
y = np.zeros((ex))
y[df['class'] == 'suicide'] = 1

sentences = sentences[:15000]
y = y[:15000]

sentences_train, sentences_test, y_train, y_test = train_test_split(
    sentences, y, test_size=0.25, random_state=1000)

vectorizer = CountVectorizer()
vectorizer.fit(sentences_train)
train_matrix = vectorizer.transform(sentences_train)
test_matrix  = vectorizer.transform(sentences_test)

'''Bernoulli Naive Bayes Classification'''
bnb = BernoulliNB(alpha=1.0, binarize=0.0, fit_prior=True, class_prior=None)
bnb = bnb.fit(train_matrix, y_train)
predictionsNB = bnb.predict(test_matrix)
scoreNB = metrics.accuracy_score(y_test, predictionsNB)
bnb_confusion_matrix = evaluation.confusion_matrix(y_test, predictionsNB)
#print(predictionsNB)
#test_df['predictionsNB'] = predictionsNB
print("Bernoulli Naive Bayes Prediction Score:", round(scoreNB*100, 2), "%")
print("Confusion Matrix:", bnb_confusion_matrix)
print("Precision:", evaluation.calc_precision(bnb_confusion_matrix))
print("Sensitivity/Positive Recall:", evaluation.calc_sensitivity(bnb_confusion_matrix))
print("Specificity/Negative Recall:", evaluation.calc_specificity(bnb_confusion_matrix))
print("F1 Score:", evaluation.calc_f1_score(bnb_confusion_matrix))

'''Gaussian Naive Bayes Classification'''
gnb = GaussianNB(var_smoothing=1e-5)
gnb = gnb.fit(train_matrix.todense(), y_train)
predictionsGNB = gnb.predict(test_matrix.todense())
scoreGNB = metrics.accuracy_score(y_test, predictionsGNB)
gnb_confusion_matrix = evaluation.confusion_matrix(y_test, predictionsGNB)
#print(predictionsGNB)
#test_df['predictionsGNB'] = predictionsGNB
print("Gaussian Naive Bayes Prediction Score:", round(scoreGNB*100, 2), "%")
print("Confusion Matrix:", gnb_confusion_matrix)
print("Precision:", evaluation.calc_precision(gnb_confusion_matrix))
print("Sensitivity/Positive Recall:", evaluation.calc_sensitivity(gnb_confusion_matrix))
print("Specificity/Negative Recall:", evaluation.calc_specificity(gnb_confusion_matrix))
print("F1 Score:", evaluation.calc_f1_score(gnb_confusion_matrix))