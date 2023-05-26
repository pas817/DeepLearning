import numpy as np

# 1-layer perceptron


def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    bias = -0.5
    temp = np.sum(x*w) + bias

    if temp <= 0:
        return 0
    else:
        return 1


def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    bias = 0.6
    temp = np.sum(x*w) + bias

    if temp <= 0:
        return 0
    else:
        return 1


def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    bias = -0.1
    temp = np.sum(x*w) + bias

    if temp <= 0:
        return 0
    else:
        return 1

# 2-layer perceptron


def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    return AND(s1, s2)


def step_function(x: np.array):
    y = x > 0
    return y.astype(int)

# 72p numpy broadcasting


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# relu는 왜 0을 기준으로 하는가?
# 음수를 왜 0으로 만드는가?
# 가중치를 학습할 때, w를 바꿔가면서 나오는 음수를 0으로 만들면 해당 입력된 부분은 예측할 부분의 음의 방향이므로, 고려하지 않는다는 뜻으로 사용할 수 있어서 그런가?


def relu(x):
    return np.maximum(0, x)


def identity_function(x):
    return x


def softmax(a):
    # C는 오버플로를 방지하기 위한 상수
    C = np.max(a)
    return np.exp(a - C) / np.sum(np.exp(a - C))
