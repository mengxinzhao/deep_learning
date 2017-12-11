import numpy as np

# Write a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.
def softmax(L):
    vote = []
    base = np.exp(L) 
    total = np.sum(base)
    votes = [base[i]/total for i in range(len(L)) ]
    return votes