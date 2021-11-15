import pandas as pd
import numpy as np
import utils
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

train_file = 'sample_3000_tokens.csv'
test_file = 'sample_test_750_tokens.csv'

train_df = pd.read_csv(train_file) 
test_df = pd.read_csv(test_file) 

train_labels = train_df['classification_int']
max_train_count = train_df['max_count'][0]
train_tokens = train_df['tokens']

test_labels = test_df['classification_int']
max_test_count = test_df['max_count'][0]
test_tokens = test_df['tokens']

padding = max(max_train_count, max_test_count)

train_matrix = utils.create_matrix(train_tokens, padding)
test_matrix = utils.create_matrix(test_tokens, padding)

# scaler = StandardScaler()
# scaled_train = scaler.fit_transform(train_matrix, train_labels)
# scaled_test = scaler.fit_transform(test_matrix, test_labels)

clf = LinearSVC(random_state=0, tol=1e-5, max_iter=100000, dual=False) # set dual to True if num_features > n_samples
# clf.fit(scaled_train, train_labels)
# predictions = clf.predict(scaled_test)
clf.fit(train_matrix, train_labels)
predictions = clf.predict(test_matrix)
print("Accuracy: ", metrics.accuracy_score(test_labels, predictions))