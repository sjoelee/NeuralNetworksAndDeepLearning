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
    
    def train(self, data, num_epochs=1, batch_size=1):
        for idx_epoch in range(num_epochs):
            n = len(data)
            batch_data = [data[i:i+batch_size] for i in xrange(0, n, batch_size)]
            for idx, batch in enumerate(batch_data):
                self.feedforward(x=batch[0][0], y=batch[0][1]) ## need to address this for the batch size
                self.backpropagate(y=batch[0][1])

    def test(self, data):
        error_rate = 0
        for test_data in data:
            x_test = test_data[0]
            y_test = test_data[1]
            self.feedforward(x_test, y_test)
            error = (y_test != self.activations[-1])
            if error:
                num_error += 1
        
        return num_error / (1.*len(data))

    def feedforward(self, x, y):
        self.activations = [x]
        self.z_inputs = [x]

        for w, b in zip(self.weights, self.biases):
            z = np.dot(w.T, self.activations[-1]) + b
            self.z_inputs.append(z)
            a = sigmoid(z)
            self.activations.append(a)
    
    def backpropagate(self, y, eta=0.3):
        # assume we have more than one layer...
        nabla_b = [np.zeros((s, 1)) for s in self.sizes[1:]]
        nabla_w = [np.zeros((s1, s2)) for s1, s2 in zip(self.sizes[:-1], self.sizes[1:])]

        # for layer L
        # note that the length of activations and z_inputs are different from nabla_b and nabla_w
        # because activations and z_inputs includes the input in the first entry
        delta = (self.activations[-1] - y) * sigmoid_prime(self.z_inputs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(self.activations[-2], delta.T)
        self.biases[-1] = self.biases[-1] - eta * nabla_b[-1]
        self.weights[-1] = self.weights[-1] - eta * nabla_w[-1]

        # for all previous layers
        for l in range(self.num_layers-1, 1, -1):
            delta = np.dot(self.weights[l-1], delta) * sigmoid_prime(self.z_inputs[l-1])
            nabla_b[l-2] = delta
            nabla_w[l-2] = np.dot(self.activations[l-2], delta.T)
            self.biases[l-2] = self.biases[l-2] - eta * nabla_b[l-2]
            self.weights[l-2] = self.weights[l-2] - eta * nabla_w[l-2]
                
#     def update_mini_batch(self, mini_batch, eta):
#         batch_size = len(mini_batch)
#         nabla_b = [np.zeros((s, 1)) for s in self.sizes[1:]]
def sigmoid(z):
    return 1./(1. + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

def cost(x, y):
    # use squared error
    return 0.5 * (np.linalg.norm(x-y) ** 2)
