import numpy as np

from network import Network

topology = [5, 3, 2]
data = np.random.rand(5)

net = Network(topology)
output = net.feedforward(data)
print(net.zeta_layer)
print(net.weights)
print(output)
