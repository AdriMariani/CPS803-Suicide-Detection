import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB

warnings.filterwarnings("ignore")

filepath = 'data/Suicide_Detection.csv'

df = pd.read_csv(filepath, header=0)
print(df.iloc[0])

sentences = df['text'].values

ex = sentences.shape[0]
y = np.zeros((ex))
y[df['class'] == 'suicide'] = 1

sentences = sentences[:1000]
y = y[:1000]

sentences_train, sentences_test, y_train, y_test = train_test_split(
    sentences, y, test_size=0.25, random_state=1000)

vectorizer = CountVectorizer()
vectorizer.fit(sentences_train)
X_train = vectorizer.transform(sentences_train)
X_test  = vectorizer.transform(sentences_test)

# classifier = BernoulliNB().fit(X_train, y_train)
# score = classifier.score(X_test, y_test)
# print('Accuracy for Bernoulli NB model: {:.4f}'.format(score))

classifier = MultinomialNB()
classifier.fit(X_train, y_train)
score = classifier.score(X_test, y_test)
print('Accuracy for Multinomial NB model: {:.4f}'.format(score))


# Indicative sentences
'''
probs = classifier.predict_log_proba(X_train)
ex = probs.shape[0]
indicators = np.array((ex))
indicators = probs[:, 0] / probs[:, 1]
dict = {}
for i in range(ex):
    dict[indicators[i]] = i

top = 25 # how many sentences

list = []
count = 0
for key in reversed(sorted(dict)):
    list.append(sentences_train[dict[key]])
    count += 1
    if count == top:
        break

print("Top {} indicative sentences are: ".format(count))
for i in range(top):
    print("{}. {}".format(i+1, list[i]))
'''

# Indicative words

X_train_tokens = vectorizer.get_feature_names()
ex = len(X_train_tokens)
zero = (classifier.feature_count_[0, :] + 1) / (classifier.class_count_[0] + ex)
one = (classifier.feature_count_[1, :] + 1) / (classifier.class_count_[1] + ex)
probs = np.log(one / zero)

dict = {}
for i in range(ex):
    dict[probs[i]] = X_train_tokens[i]

top = 25 # how many words

list = []
count = 0
for key in reversed(sorted(dict)):
    list.append(dict[key])
    count += 1
    if count == top:
        break

print("Top {} indicative words are: {}".format(top, list))

list = []
count = 0
for key in sorted(dict):
    list.append(dict[key])
    count += 1
    if count == top:
        break

print("Top {} non-indicative words are: {}".format(top, list))