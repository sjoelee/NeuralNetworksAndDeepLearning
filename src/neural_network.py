import numpy as np

class NeuralNetwork:
    def __init__(self, sizes):
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.biases = [np.random.randn(s, 1) for s in sizes[1:]]
        self.weights = [np.random.randn(s1, s2) for s1, s2 in zip(sizes[:-1], sizes[1:])]
        self.activations = []
        self.z_inputs = []
        
        np.random.seed(2015)
    
    def feedforward(self, x, y):
        self.activations = [x]
        self.z_inputs = [x]

        for w, b in zip(self.weights, self.biases):
            z = np.dot(w.T, self.activations[-1]) + b
            self.z_inputs.append(z)
            a = sigmoid(z)
            self.activations.append(a)
    
    def backpropagate(self, y, eta=0.3):
        # backpropagate
        # for layer L
        # assume we have more than one layer...
        nabla_b = [np.zeros((s, 1)) for s in self.sizes[1:]]
        nabla_w = [np.zeros((s1, s2)) for s1, s2 in zip(self.sizes[:-1], self.sizes[1:])]

        #initialize (for layer L = 2)
        # note that the length of activations and z_inputs are different from nabla_b and nabla_w
        # because activations and z_inputs includes the input
        delta = (self.activations[-1] - y) * sigmoid_prime(self.z_inputs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(self.activations[-2], delta.T)
        self.biases[-1] = self.biases[-1] - eta * nabla_b[-1]
        self.weights[-1] = self.weights[-1] - eta * nabla_w[-1]

        # for all previous layers
        for l in range(num_layers-1, 1, -1):
            delta = np.dot(self.weights[l-1], delta) * sigmoid_prime(self.z_inputs[l-1])
            nabla_b[l-2] = delta
            nabla_w[l-2] = np.dot(self.activations[l-2], delta.T)
            self.biases[l-2] = self.biases[l-2] - eta * nabla_b[l-2]
            self.weights[l-2] = self.weights[l-2] - eta * nabla_w[l-2]
