import numpy as np

def create_matrix(tokens, padding):
    n = len(tokens)
    matrix = np.zeros(shape=(n,padding),dtype='object')
    for i in range(0, n):
        value = (tokens[i])
        value = value.replace("[", "")
        value = value.replace("]", "") 
        value = value.split(", ")
        #print(len(value), value)
        for j in range(0, len(value)):
            matrix[i][j] = int(value[j])
    return matrix