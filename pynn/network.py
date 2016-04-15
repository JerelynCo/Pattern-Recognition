import numpy as np


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
            assert weights.size == len(topology) - 1, "weights size does not match topology"
            self.weights = weights
        # zeta values are the outputs values of a layer prior to being subjected
        # to the activation function. This is used for backpropagation
        # Input data is not part of zeta
        self.zeta_layer = []

    @staticmethod
    def activation(zeta):
        return np.tanh(zeta)

    def feedforward(self, input_data):
        assert input_data.size == self.topology[0], "input size exceeds number of input nodes"
        self.zeta_layer = []
        alpha = input_data
        for layer_weights in self.weights:
            zeta = np.dot(layer_weights, alpha)
            self.zeta_layer.append(zeta)
            alpha = self.activation(zeta)
        return alpha

    def compute_cost(self, zeta, target):
        assert len(target) == self.topology[-1], "target size exceeds number of output nodes."
        target = np.array(target)
        return np.sum(np.square(zeta - target))
