import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
import sys
sys.path.append('..')
import utils.utils as utils
import utils.evaluation as evaluation
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

path = '../datasets/c_Suicide_Detection.csv'
df = pd.read_csv(path, header=0)

sentences = df['notesCleaned'].values

ex = sentences.shape[0]
y = np.zeros((ex))
y[df['class'] == 'suicide'] = 1

sentences = sentences[:50000]
y = y[:50000]

sentences_train, sentences_test, y_train, y_test = train_test_split(
    sentences, y, test_size=0.25, random_state=1000)

vectorizer = CountVectorizer()
vectorizer.fit(sentences_train)
train_matrix = vectorizer.transform(sentences_train)
test_matrix  = vectorizer.transform(sentences_test)

'''Classification'''
clf = LogisticRegression(max_iter=1200000)
clf.fit(train_matrix, y_train)
predictions = clf.predict(test_matrix)
confusion_matrix = evaluation.confusion_matrix(y_test, predictions)

print("Accuracy:", metrics.accuracy_score(y_test, predictions))
print("Confusion Matrix:", confusion_matrix)
print("Precision:", evaluation.calc_precision(confusion_matrix))
print("Sensitivity/Positive Recall:", evaluation.calc_sensitivity(confusion_matrix))
print("Specificity/Negative Recall:", evaluation.calc_specificity(confusion_matrix))
print("F1 Score:", evaluation.calc_f1_score(confusion_matrix))
