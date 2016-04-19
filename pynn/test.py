import numpy as np

from network import Network

topology = [5, 4, 2]

data = np.array(np.random.random([2, 5]))

target = [[0, 1],
          [1, 0]]

epochs = 500
eta = 0.5

net = Network(topology)

for epoch in range(epochs):
    output = net.feedforward(data[0])
    net.backprop(eta, target[0])
    output1 = net.feedforward(data[1])
    net.backprop(eta, target[1])
    print("Epoch {0}: {1}".format(epoch, net.compute_cost(output, target)))
    print("Epoch {0} Output: {1} ".format(epoch, output))
    print("Epoch {0}: {1}".format(epoch, net.compute_cost(output1, target)))
    print("Epoch {0} Output: {1} ".format(epoch, output1))
    #
    # output = net.feedforward(data)
    # net.backprop(eta, target)
    # print(net.weights)
