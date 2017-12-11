import numpy as np

# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.
#
def cross_entropy(Y, P):
    """
    return: -sum(y *ln(p) + (1-y)*ln(1-p))
    """
    if len(Y)  == len(P):
        element = [Y[i] * np.log(P[i]) + (1-Y[i]) * np.log(1-P[i]) for i in range(len(Y))]
        return -np.sum(element)
    else:
        print("dimensions of Y and P are not the same!")
        return -np.inf
        
