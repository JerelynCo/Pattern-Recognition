import numpy as np
from network import Network

topology = [600, 10, 2]
data = np.random.rand(600)

net = Network(topology)
output = net.feedforward(data)
print(output)
print(net.computeCost(output, [0.2, -0.3]))