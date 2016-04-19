import numpy as np

from network import Network

topology = [5, 4, 2]

data = np.array([0.5, -0.23, 0.45, -0.123, 0.55])
target = [0, 1]

epochs = 500
eta = 0.10

net = Network(topology)
output = net.feedforward(data)

for epoch in range(epochs):
    output = net.feedforward(data)
    net.backprop(eta, target)
    print("Epoch {0}: {1}".format(epoch, net.compute_cost(output, target)))
    print("Epoch {0} Output: {1} ".format(epoch, output))
    #
    # output = net.feedforward(data)
    # net.backprop(eta, target)
    # print(net.weights)
