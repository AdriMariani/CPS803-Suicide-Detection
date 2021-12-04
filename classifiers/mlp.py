import pandas as pd
import sys
sys.path.append('..')
import utils.utils as utils
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

'''Import Training Data'''
train_name = 'Suicide_Detection_tokens1'
train_df = pd.read_csv(train_name+'.csv') 
train_df = train_df.loc[train_df['limit'] == 0] #filter records where tokens <512

'''Import Test Data'''
test_name = 'suicide_notes_tokens1'
test_df = pd.read_csv(test_name+'.csv') 

'''Dataframes'''
train_labels = train_df['classification_int']
train_text = test_df['notesCleaned']
max_train_count = train_df['max_count'][0]
train_tokens = train_df['tokens']

test_labels = test_df['classification_int']
test_text = test_df['notesCleaned']
max_test_count = test_df['max_count'][0]
test_tokens = test_df['tokens']

'''Constants'''
if max_train_count > 512 or max_test_count:
    padding = 512
else:
    padding = max(max_train_count, max_test_count)

train_matrix = utils.create_matrix(train_tokens, padding)
test_matrix = utils.create_matrix(test_tokens, padding)

mlp_clf = MLPClassifier()
mlp_clf.fit(train_matrix, train_labels)
mlp_clf.fit(train_matrix, train_labels)
predictions = mlp_clf.predict(test_matrix)

'''MLP Classification'''
print(test_name)
test_df['predictionsLR'] = predictions
print('Accuracy: {:.2f}%'.format((accuracy_score(test_labels, predictions))*100))
test_df.to_csv(test_name+'_predictions.csv', index=False)