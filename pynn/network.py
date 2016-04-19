import random

import numpy as np


class Network:
    # Example topology is [3,5,1] with input having 3 neurons
    # hidden layer with 5, output wuth 1
    def __init__(self, topology, weights=np.array([])):
        self.topology = topology
        if (weights.size == 0):
            # [5 x 3], [1 x 5]
            dims = list(zip(topology[1:], topology[:-1]))
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

    @staticmethod
    def activation(zeta):
        return np.tanh(zeta)

    @staticmethod
    def activation_prime(zeta):
        return 1 - np.pow(zeta, 2)

    def feedforward(self, input_data):
        assert input_data.size == self.topology[
            0], "input size exceeds number of input nodes"
        self.zeta_layers = []
        self.activation_layers = [input_data]
        alpha = input_data
        for layer_weights in self.weights:
            zeta = np.dot(layer_weights, alpha)
            self.zeta_layers.append(zeta)

            alpha = self.activation(zeta)
            self.activation_layers.append(alpha)
        return alpha

    def update_weights(self, input_data, target, eta):
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        delta_nabla_w = self.backprop(input_data, target)
        nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [
            w - (eta / len(input_data)) * nw for w, nw in zip(self.weights, nabla_w)]

    def backprop(self, input_data, target):
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        delta = self.cost_derivative(
            self.activation_layers[-1], target) * self.activation_prime(
            self.zeta_layers[-1])
        nabla_w[-1] = np.dot(delta, np.transpose(self.activation_layers[-1])) # theoretically, should be self.activation_layers[-2] to match equation

        for l in range(2, len(self.topology)):
            # From the second to the first layer backwards
            delta = np.dot(np.transpose(
                self.weights[-l + 1]), delta) * self.activation_prime(self.zeta_layers[-l])
            nabla_w[-l] = np.dot(
                delta, np.transpose(self.activation_layers[-l])) # theoretically, should be self.activation_layers[-l-1]) to match equation
        return nabla_w

    def compute_cost(self, activation, target):
        assert len(
            target) == self.topology[-1], "target size exceeds number of output nodes."
        target = np.array(target)
        return np.sum(np.square(activation - target)) / 2

    def cost_derivative(self, activation, target):
        # Partial derivatives for the output activations
        return (target - activation)
