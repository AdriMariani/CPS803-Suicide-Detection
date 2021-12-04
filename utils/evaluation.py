def get_indicative_words(x, y):
    '''Returns list of top 10 most indicative words
        
        x: test inputs 
        y: test predictions
    '''
    # this is the way tony said to do it but it doesn't work too well
    # will ask him tomorrow
    # n = len(x)
    # word_dict = {}
    # for i in range(n):
    #     words = x[i].split()
    #     for word in words:
    #         if not word in word_dict:
    #             if y[i] == 1:
    #                 word_dict[word] = [1, 1]
    #             else:
    #                 word_dict[word] = [0, 1]
    #         else:
    #             word_dict[word][1] += 1
    #             if y[i] == 1:
    #                 word_dict[word][0] += 1
    
    # indicative_words = {k: v[0] / v[1] for k,v in word_dict.items()}
    # return {k: v for k,v in sorted(indicative_words.items(), key=lambda word: word[1], reverse=True)}

    # this method just counts frequencies of words appearing in suicide examples
    n = len(x)
    word_dict = {}
    stop_words = {'i', 'and', 'to', 'the', 'a', 'my', 'it', 'of', 't', 'but', 'that', 'm', 'be', 'just', 'for',
        'do', 'in', 'is', 's', 'was', 'so', 'what', 'im', 'on', 'like', 'don', 'with', 'all', 'amp', 'at', 'if', 'as',
        'x200b'
    }
    for i in range(n):
        if y[i] == 1:
            words = x[i].split()
            for word in words:
                if not word in stop_words:
                    if not word in word_dict:
                        word_dict[word] = 1
                    else:
                        word_dict[word] +=  1
    
    return {k: v for k,v in sorted(word_dict.items(), key=lambda word: word[1], reverse=True)}

def confusion_matrix(true_labels, predictions):
    '''Returns the number of true positives, true negatives, false positives,
    and false negatives
        
        true_labels: true labels of data
        predictions: predicted labels of data
    '''
    FP = 0
    FN = 0
    TP = 0
    TN = 0
    for i in range(len(true_labels)):
        if predictions[i] == 1 and true_labels[i] == 0:
            FP += 1
        elif predictions[i] == 1 and true_labels[i] == 1:
            TP += 1
        elif predictions[i] == 0 and true_labels[i] == 0:
            TN += 1
        else:
            FN += 1
    return {'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN}

def calc_precision(confusion_matrix):
    """Calculates the precision of the model

        confusion_matrix: Confusion matrix of predictions
    """
    return confusion_matrix['TP'] / (confusion_matrix['TP'] + confusion_matrix['FP'])

def calc_sensitivity(confusion_matrix):
    """Calculates the sensitivity/positive recall of the model

        confusion_matrix: Confusion matrix of predictions
    """
    return confusion_matrix['TP'] / (confusion_matrix['TP'] + confusion_matrix['FN'])

def calc_specificity(confusion_matrix):
    """Calculates the specificity/negative recall of the model

        confusion_matrix: Confusion matrix of predictions
    """
    return confusion_matrix['TN'] / (confusion_matrix['TN'] + confusion_matrix['FP'])

def calc_f1_score(confusion_matrix):
    """Calculates the F1 score of the model

        confusion_matrix: Confusion matrix of predictions
    """
    precision = calc_precision(confusion_matrix)
    recall = calc_sensitivity(confusion_matrix)
    return 2 * precision * recall / (precision + recall)