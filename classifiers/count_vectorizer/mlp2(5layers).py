import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB

warnings.filterwarnings("ignore")

filepath = 'data/bert_test_data	.csv'

df = pd.read_csv(filepath, header=0)
df = df.dropna()

sentences = df['text'].values

ex = sentences.shape[0]
y = np.zeros((ex))
y[df['class'] == 'suicide'] = 1



sentences_train, sentences_test, y_train, y_test = train_test_split(
    sentences, y, test_size=0.25, random_state=1000)

vectorizer = CountVectorizer()
vectorizer.fit(sentences_train)
X_train = vectorizer.transform(sentences_train)
X_test  = vectorizer.transform(sentences_test)


from keras.models import Sequential
from keras import layers

input_dim = X_train.shape[1]  # Number of features

classifier = Sequential()
classifier.add(layers.Dense(100, input_dim=input_dim, activation='relu'))
classifier.add(layers.Dropout(0.2))
classifier.add(layers.Dense(10, activation='relu'))
classifier.add(layers.Dropout(0.2))
classifier.add(layers.Dense(1, activation='sigmoid'))

classifier.compile(loss='binary_crossentropy', 
               optimizer='adam', 
               metrics=['accuracy'])
               
history = classifier.fit(X_train, y_train,
                    epochs=100,
                    verbose=False,
                    validation_data=(X_test, y_test),
                    batch_size=10)


loss, accuracy = classifier.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = classifier.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

out = classifier.predict(X_test)

out = np.squeeze(out,axis=1)
out = [0 if a<0.5 else 1 for a in out]


from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, out))


