import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from joblib import dump, load

path = '../../datasets/c_Suicide_Detection.csv'
test_set2_path = '../../datasets/c_reddit_depression_suicidewatch.csv'
df = pd.read_csv(path, header=0)
df2 = pd.read_csv(test_set2_path, header=0)

suicide_watch_sentences = df2['notesCleaned'].values
suicide_watch_labels = np.zeros(suicide_watch_sentences.shape[0])
suicide_watch_labels[df2['class'] == 'suicide'] = 1

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
suicide_watch_matrix = vectorizer.transform(suicide_watch_sentences)

load_model = True  # if model is saved to file set True

if not load_model:
    clf = LinearSVC(random_state=0, tol=1e-5, max_iter=100000, dual=False, C=0.1) # set dual to True if num_features > n_samples
    clf.fit(train_matrix, y_train)
    dump(clf, 'svm.joblib')
else:
    clf = load('svm.joblib')

predictions = clf.predict(test_matrix)

print("Accuracy:", metrics.accuracy_score(y_test, predictions))
print("Confusion_matrix: \n{}".format(metrics.confusion_matrix(y_test, predictions)))
print("Precision: {:.4f}".format(metrics.precision_score(y_test, predictions)))
print("Recall: {:.4f}".format(metrics.recall_score(y_test, predictions)))
print("F1 Score: {:.4f}".format(metrics.f1_score(y_test, predictions)))
# print("Top 10 indicative words of suicide:", evaluation.get_indicative_words(sentences_test, predictions))

suicide_watch_predictions = clf.predict(suicide_watch_matrix)

print("Suicide Watch Set Accuracy:", metrics.accuracy_score(suicide_watch_labels, suicide_watch_predictions))
print("Confusion_matrix: \n{}".format(metrics.confusion_matrix(suicide_watch_labels, suicide_watch_predictions)))
print("Precision: {:.4f}".format(metrics.precision_score(suicide_watch_labels, suicide_watch_predictions)))
print("Recall: {:.4f}".format(metrics.recall_score(suicide_watch_labels, suicide_watch_predictions)))
print("F1 Score: {:.4f}".format(metrics.f1_score(suicide_watch_labels, suicide_watch_predictions)))