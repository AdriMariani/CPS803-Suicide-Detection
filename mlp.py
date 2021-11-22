import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report

train_file = 'sample_3000_tokens.csv'
test_file = 'sample_test_750_tokens.csv'

train_df = pd.read_csv(train_file) 
test_df = pd.read_csv(test_file) 

train_labels = train_df['classification_int']
max_train_count = train_df['max_count'][0]
train_tokens = train_df['tokens']

test_labels = test_df['classification_int']
test_text = test_df['notesCleaned']
max_test_count = test_df['max_count'][0]
test_tokens = test_df['tokens']

padding = max(max_train_count, max_test_count)

train_matrix = utils.create_matrix(train_tokens, padding)
test_matrix = utils.create_matrix(test_tokens, padding)

mlp_clf = MLPClassifier(hidden_layer_sizes=(5,2), max_iter = 300,activation = 'relu', solver = 'adam')
                        
mlp_clf.fit(train_matrix, train_labels)
predictions = mlp_clf.predict(test_matrix)
confusion_matrix = evaluation.confusion_matrix(test_labels, predictions)

print("Accuracy:", metrics.accuracy_score(test_labels, predictions))
print("Confusion Matrix:", confusion_matrix)
print("Precision:", evaluation.calc_precision(confusion_matrix))
print("Sensitivity/Positive Recall:", evaluation.calc_sensitivity(confusion_matrix))
print("Specificity/Negative Recall:", evaluation.calc_specificity(confusion_matrix))
print("F1 Score:", evaluation.calc_f1_score(confusion_matrix))
print("Top 10 indicative words of suicide:", evaluation.get_indicative_words(test_text, predictions))
