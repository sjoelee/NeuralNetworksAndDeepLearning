from neural_network import NeuralNetwork

import mnist_loader

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

sizes = [784, 30, 10]
nn = NeuralNetwork(sizes)
nn.train(training_data)
nn.test(test_data)
