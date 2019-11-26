import numpy as np

def sigmoid(x):
    return 1/1 + np.exp(-x)

def a_prime(x):
    return  sigmoid(x)

w = 2
a = 0.78
b = -2
a_p = w*a + b

print(sigmoid(a_p))
print(a_prime(a_p))