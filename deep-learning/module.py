import numpy as np

# 1-layer perceptron
def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    bias = -0.5
    temp = np.sum(x*w) + bias

    if temp <= 0: return 0
    else:         return 1

def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    bias = 0.6
    temp = np.sum(x*w) + bias

    if temp <= 0: return 0
    else:         return 1

def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    bias = -0.1
    temp = np.sum(x*w) + bias

    if temp <= 0: return 0
    else:         return 1

# 2-layer perceptron
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    return AND(s1, s2)