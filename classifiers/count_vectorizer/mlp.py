import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from joblib import dump, load

filepath = '../../datasets/c_Suicide_Detection.csv'
suicide_notes_path = '../../datasets/c_suicide_notes.csv'

df = pd.read_csv(filepath, header=0)
df2 = pd.read_csv(suicide_notes_path, header=0)

suicide_notes_sentences = df2['notesCleaned'].values
nan_values = pd.isnull(suicide_notes_sentences)
suicide_notes_sentences = suicide_notes_sentences[~ nan_values]
suicide_notes_labels = np.ones(suicide_notes_sentences.shape[0])

sentences = df['notesCleaned'].values

ex = sentences.shape[0]
y = np.zeros((ex))
y[df['class'] == 'suicide'] = 1

# sentences = sentences[:5000]
# y = y[:5000]

sentences_train, sentences_test, y_train, y_test = train_test_split(
    sentences, y, test_size=0.25, random_state=1000)

vectorizer = CountVectorizer()
vectorizer.fit(sentences_train)
X_train = vectorizer.transform(sentences_train)
X_test  = vectorizer.transform(sentences_test)
suicide_notes_vector = vectorizer.transform(suicide_notes_sentences)

load_model = False  # if model is saved to file set True

if not load_model:
  mlp_clf = MLPClassifier()
  mlp_clf = mlp_clf.fit(X_train, y_train)
  dump(mlp_clf, 'mlp.joblib')
else:
  mlp_clf = load('mlp.joblib')

predictionsMLP = mlp_clf.predict(X_test)

print("Accuracy:", metrics.accuracy_score(y_test, predictionsMLP))
print("Confusion_matrix: \n{}".format(metrics.confusion_matrix(y_test, predictionsMLP)))
print("Precision: {:.4f}".format(metrics.precision_score(y_test, predictionsMLP)))
print("Recall: {:.4f}".format(metrics.recall_score(y_test, predictionsMLP)))
print("F1 Score: {:.4f}".format(metrics.f1_score(y_test, predictionsMLP)))

notes_predictions = mlp_clf.predict(suicide_notes_vector)
print("Suicide Notes Set Accuracy:", metrics.accuracy_score(suicide_notes_labels, notes_predictions))
print("Confusion_matrix: \n{}".format(metrics.confusion_matrix(suicide_notes_labels, notes_predictions)))
print("Precision: {:.4f}".format(metrics.precision_score(suicide_notes_labels, notes_predictions)))
print("Recall: {:.4f}".format(metrics.recall_score(suicide_notes_labels, notes_predictions)))
print("F1 Score: {:.4f}".format(metrics.f1_score(suicide_notes_labels, notes_predictions)))