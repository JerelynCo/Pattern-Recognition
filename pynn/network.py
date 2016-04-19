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
            assert weights.size == len(
                topology) - 1, "weights size does not match topology"
            self.weights = weights

    @staticmethod
    def activation(zeta):
        return np.tanh(zeta)

    @staticmethod
    def activation_prime(zeta):
        return 1.0 / np.square(zeta)


    def feedforward(self, input_data):
        assert input_data.size == self.topology[
            0], "input size exceeds number of input nodes"
        self.zeta_layers = []
        self.alpha_layers = [input_data]
        alpha = input_data
        for layer_weights in self.weights:
            zeta = np.dot(layer_weights, alpha)
            self.zeta_layers.append(zeta)

            alpha = self.activation(zeta)
            self.alpha_layers.append(alpha)
        return alpha

    def backprop(self, eta, target):
        individual_delta_layers = []
        delta_layers = []
        delta_nabla_weights = []

        individual_delta_layers.insert(0, (self.cost_derivative(self.alpha_layers[-1], target) * self.activation_prime(
            self.zeta_layers[-1])))
        for l in range(2, len(self.topology)):
            individual_delta_layers.insert(0, np.dot(self.weights[-l + 1].T,
                                                     individual_delta_layers[-l + 1]) * self.activation_prime(
                self.zeta_layers[-l]))

        counter = 0
        for i in individual_delta_layers:
            delta_layers.append(i.sum() / self.topology[counter])
            counter += 1

        delta_nabla_weights.insert(0, np.dot(delta_layers[-1], self.alpha_layers[-2]))
        for l in range(2, len(self.topology)):
            delta_nabla_weights.insert(0, np.dot(delta_layers[-l], self.alpha_layers[-l - 1]))

        self.weights = [w - eta * nw for w, nw in zip(self.weights, delta_nabla_weights)]

        # for d, a in zip(self.delta_layers, self.alpha_layers):
        #     np.dot(a.T, d)


    def compute_cost(self, zeta, target):
        assert len(
            target) == self.topology[-1], "target size exceeds number of output nodes."
        target = np.array(target)
        return np.sum(np.square(zeta - target))

    def cost_derivative(self, output_activations, target):
        # Partial derivatives for the output activations
        return (output_activations - target)
