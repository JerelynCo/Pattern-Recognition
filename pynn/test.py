import numpy as np
from network import Network

topology = [5, 3, 2]

data = np.array([0.5, -0.23, 0.45, -0.123, 0.55])
target = [1, 2]

epochs = 5	
eta = 0.15

net = Network(topology)
output = net.feedforward(data)

for epoch in range(epochs):
	output = net.feedforward(data)
	net.update_weights(data, target, eta)
	print("Epoch {0}: {1}".format(epoch, net.compute_cost(output, target)))
	print("Weights: {0}".format(net.weights))

