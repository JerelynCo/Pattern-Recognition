import numpy as np


class Network:
    # Example topology is [3,5,1] with input having 3 neurons
    # hidden layer with 5, output wuth 1
    def __init__(self, topology, weights=[]):
        self.topology = topology
        # [5 x 3], [1 x 5]
        dims = list(zip(topology[1:], topology))
        # weights[between what layers][neuron_i][neuron_j]
        self.weights = np.array([np.random.random(i) for i in dims
                                 ])

    def activation(self, zeta):
        return np.tanh(zeta)

    def feedforward(self, input_data):
        assert input_data.size == self.topology[0], "input size exceeds number of input nodes"
        zeta = input_data
        for layer_weights in self.weights:
            zeta = self.activation(np.dot(layer_weights, zeta))
        return zeta

    def computeCost(self, zeta, target):
        assert len(target) == self.topology[-1], "target size exceeds number of output nodes."
        target = np.array(target)
        return np.sum(np.square(zeta - target))