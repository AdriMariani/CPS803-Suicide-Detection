import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

filepath = 'data/Suicide_Detection.csv'

df = pd.read_csv(filepath, header=0)
print(df.iloc[0])

sentences = df['text'].values

ex = sentences.shape[0]
y = np.zeros((ex))
y[df['class'] == 'suicide'] = 1

sentences = sentences[:5000]
y = y[:5000]

sentences_train, sentences_test, y_train, y_test = train_test_split(
    sentences, y, test_size=0.25, random_state=1000)

vectorizer = CountVectorizer()
vectorizer.fit(sentences_train)
X_train = vectorizer.transform(sentences_train)

X_test  = vectorizer.transform(sentences_test)

classifier = LogisticRegression(max_iter=100, solver='newton-cg')
classifier.fit(X_train, y_train)
score = classifier.score(X_test, y_test)
print('Accuracy for data: {:.4f}'.format(score))