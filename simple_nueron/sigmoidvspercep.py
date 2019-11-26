import numpy as np

def sigmoid(x):
    # Our activation function: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))

class Percep:
    def __init__(self):
        self.c = 2
        self.w1 = 0  * self.c
        self.w2 = 1 * self.c
        self.b1 = 0 * self.c


    def feedforward(self, x):
        h1 = self.w1 * x + self.b1
        o1 = self.w2 * h1
        if o1 > 0:
            o1 = 1
        elif o1 <= 0:
            o1 = 0
        print("h1 Percep: %.3f" %(h1))
        return o1


class Neural:
    def __init__(self):
        self.c = 2
        self.w1 = 0 * self.c
        self.w2 = 1 * self.c
        self.b1 = 0 * self.c

    def feedforward(self, x):
        h1 = sigmoid(self.w1 * x + self.b1)
        o1 = sigmoid(self.w2 * h1)
        print("h1 Sigmoid: %.3f" %(h1))
        return o1

neural = Neural()
percep = Percep()
z = neural.feedforward(2)
y = percep.feedforward(2)

print("Sigmoid : %.3f" %(z))
print("Percep : %.3f" %(y))