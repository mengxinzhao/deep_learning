#!/usr/bin/env python
import numpy as np

# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.
#
def cross_entropy(Y, P):
    """
    return: -sum(y *ln(p) + (1-y)*ln(1-p))
            cross-entropy is inversely proportional to the total probability of an outcome
    """
    if len(Y)  == len(P):
        element = [Y[i] * np.log(P[i]) + (1-Y[i]) * np.log(1-P[i]) for i in range(len(Y))]
        return -np.sum(element)
    else:
        print("dimensions of Y and P are not the same!")
        return -np.inf
        

def multi_class_cross_entropy(Y,P):
    """
    param: 
    Y: n x m multidimensional indicator
    P: n x m multidimensional probability 
    return: - sum over i ( sum  over j  for Yij * ln(Pij))

    """
    Y = np.array(Y)
    P = np.array(P)
    rows = []
    if Y.shape == P.shape:
        for row in range(Y.shape[0]):
            rows.append( [Y[row][j] * np.log(P[row][j]) + (1-Y[row][j]) * np.log(1-P[row][j]) for j in range(Y.shape[1]) ] )
        return -np.sum(rows)
    else:
        print("dimensions of Y and P are not the same!")
        return -np.inf


if __name__ == "__main__":
    Y=[1,0,1,1] 
    P=[0.4,0.6,0.1,0.5]

    print("{:.4f}".format(cross_entropy(Y,P)))

    Y=[[1,0,1], [0,1,1], [1,1,0]]
    P=[ [0.7, 0.3, 0.1], [0.2,0.4,0.5], [0.1,0.3,0.4] ]
    print("{:.4f}".format(multi_class_cross_entropy(Y,P)))