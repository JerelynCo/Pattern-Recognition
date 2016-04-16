import numpy as np
import random

class Network:
    # Example topology is [3,5,1] with input having 3 neurons
    # hidden layer with 5, output wuth 1
    def __init__(self, topology, weights=np.array([])):
        self.topology = topology
        if (weights.size == 0):
            # [5 x 3], [1 x 5]
            dims = list(zip(topology[1:], topology))
            # weights[between what layers][neuron_i][neuron_j]
            self.weights = np.array([np.random.random(i) for i in dims
                                     ])
        else:
            assert weights.size == len(
                topology) - 1, "weights size does not match topology"
            self.weights = weights
        # zeta values are the outputs values of a layer prior to being subjected
        # to the activation function. This is used for backpropagation
        # Input data is not part of zeta
        self.zeta_layer = []

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        # Training data: list of tuples (x = inputs, y=target)
        # Test data: if provided, network will be evaluated against it
        for epoch in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[index:index+mini_batch_size] for index in range(0, len(test_data), mini_batch_size)]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            if test_data:
                print("Epoch {0}: {1}/{2}".format(epoch, self.evaluate(test_data), len(test_data)))
            
            else:
                print("Epoch {0} complete".format(epoch))

    def update_mini_batch(self, mini_batch, eta):
        # Updates weights by backprop per batch
        for x, y in mini_batch:
            nabla_w = [np.zeros(w.shape) for w in self.weights]
            delta_nabla_w = self.backprop(x,y)
            nabla_w = [nw + dnw for nw, dnw iin zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw for w,nw in zip(self.weights,nabla_w)]

    def evaluate(self, test_data):
        # Classification: resolved by getting the highest output value's index
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    @staticmethod
    def activation(zeta):
        return np.tanh(zeta)

    @staticmethod
    def activation_prime(zeta):
        return 1 / (np.cosh(zeta) ** 2)

    def feedforward(self, input_data):
        assert input_data.size == self.topology[
            0], "input size exceeds number of input nodes"
        self.zeta_layers = []
        self.activations = [input_data]
        alpha = input_data
        for layer_weights in self.weights:
            zeta = np.dot(layer_weights, alpha)
            self.zeta_layers.append(zeta)

            alpha = self.activation(zeta)
            self.activations_layer.append(alpha)
        return alpha

    def backprop(self, input_data, target):
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        delta = self.cost_derivative(
            self.activations_layer[-1], target) * self.activation_prime[self.zeta_layers[-1]]
        nabla_w = np.dot(delta, self.activations_layer[-2].transpose())

        for l in range(2, len(self.topology)):
            # From the second to the last layer backwards
            delta = np.dot(
                self.weights[-l + 1].transpose, delta) * activation_prime(self.zeta_layers-l])
            nabla_w = np.dot(delta, activations[-l-1].transpose())
        return nabla_w


    def compute_cost(self, zeta, target):
        assert len(
            target) == self.topology[-1], "target size exceeds number of output nodes."
        target = np.array(target)
        return np.sum(np.square(zeta - target))

    def cost_derivative(self, output_activations, target):
        # Partial derivatives for the output activations
        return (output_activations - target)
