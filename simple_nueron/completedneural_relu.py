# Supervised learning. Thank for all code from https://victorzhou.com/blog/intro-to-neural-networks/
# Try Using ReLu as activation function
# Relu seems to make neuron dead so try leeky Relu!
# Leaky Relu has some error about overflow in double_scalars ugh. If it work It work good though

import numpy as np


def sigmoid(x):
    # Our activation function: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))

def derive_sigmoid(x):
    # Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
    fx = sigmoid(x)
    return fx * (1 - fx)

def mse_loss(y_true, y_pred):
    # y_true and y_pred are numpy arrays of the same length.
    return ((y_true - y_pred) ** 2).mean()

# Normal Relu
def relu(x):
    return np.maximum(0, x)

def derive_relu(x):
    if x >= 0:
        return 1
    elif x < 0:
        return 0

#Leaky ReLu
def leakyrelu(x):
    if x >= 0:
        return x
    elif x < 0:
        return 0.01 * x

def derive_leakyrelu(x):
    if x >= 0:
        return 1
    elif x < 0:
        return 0.01

class NeuralNetwork:
    '''
    A neural network with:
        - 2 inputs
        - a hidden layer with 2 neurons (h1, h2)
        - an output layer with 1 neuron (o1)
    Each neuron has the same weights and bias:
        - w = [0, 1]
        - b = 0

    *** DISCLAIMER ***:
    The code below is intended to be simple and educational, NOT optimal.
    Real neural net code looks nothing like this. DO NOT use this code.
    Instead, read/run it to understand how this specific network works.
    '''


    def __init__(self):
        # Weights
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()

        # Bias
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()


    def feedforward(self, x):
        # x is a numpy array with 2 elements.
        h1 = leakyrelu(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        h2 = leakyrelu(self.w3 * x[0] + self.w4 * x[1] + self.b1)
        o1 = leakyrelu(self.w5 * h1 + self.w6 * h2 + self.b3)
        return o1

    def train(self, data, all_y_trues):
        '''
        - data is a (n x 2) numpy array, n = # of samples in the dataset.
        - all_y_trues is a numpy array with n elements.
          Elements in all_y_trues correspond to those in data.
        '''
        learn_rate = 0.1
        epochs = 1000 # number of times to loop through the entire dataset

        for epochs in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                # --- Do a feedforward (we'll need these values later)
                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
                h1 = leakyrelu(sum_h1)

                sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
                h2 = leakyrelu(sum_h2)

                sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
                o1 = leakyrelu(sum_o1)
                y_pred = o1

                # --- Calculate partial derivatives.
                # --- Naming: d_L_d_w1 represents "partial L / partial w1"
                d_L_d_ypred = -2 * (y_true - y_pred)

                # Neuron o1
                d_ypred_d_w5 = h1 * derive_leakyrelu(sum_o1)
                d_ypred_d_w6 = h2 * derive_leakyrelu(sum_o1)
                d_ypred_d_b3 = derive_leakyrelu(sum_o1)

                d_ypred_d_h1 =  self.w5 * derive_leakyrelu(sum_o1)
                d_ypred_d_h2 =  self.w6 * derive_leakyrelu(sum_o1)

                # Neuron h1
                d_h1_d_w1 = x[0] * derive_leakyrelu(sum_h1)
                d_h1_d_w2 = x[1] * derive_leakyrelu(sum_h1)
                d_h1_d_b1 = derive_leakyrelu(sum_h1)

                # Neuron h2
                d_h2_d_w3 = x[0] * derive_leakyrelu(sum_h2)
                d_h2_d_w4 = x[1] * derive_leakyrelu(sum_h2)
                d_h2_d_b2 = derive_leakyrelu(sum_h2)

                # --- Update weights and biases
                # Neuron h1
                self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

                # Neuron h2 
                self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
                self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
                self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

                # Neuron o1
                self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5
                self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6
                self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3
                
                # --- Calculate total loss at the end of each epoch
                if epochs % 10 == 0 :
                    y_pred = np.apply_along_axis(self.feedforward, 1 , data)
                    loss = mse_loss(all_y_trues, y_pred)
                    print("Epoch %d loss: %.3f" % (epochs, loss))

# Define dataset
data = np.array([
    [2, -1],  # Alice
    [25, 6],  # Bob
    [17, 4],  # Charlie
    [-15, -6] # Diana
])
all_y_trues = np.array([
    1, # Alice
    0, # Bob
    0, # Charlie
    1 # Diana
])

# Train our neural network!
network = NeuralNetwork()
network.train(data, all_y_trues)

# Make some predictions
emily = np.array([-7, -3]) # 128 pounds, 63 inches
frank = np.array([20 , 2]) # 155 pounds, 68 inches
print("Emily: %.3f" %network.feedforward(emily)) # 0.984 F
print("Frank: %.3f" %network.feedforward(frank)) # 0.048 M 