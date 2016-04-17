import numpy as np

from network import Network

topology = [5, 2]

data = np.array([0.2, -0.4, 0.85, -0.47, 0.3])
target = np.array([1, 2])

net = Network(topology)
output = net.feedforward(data)
print(net.zeta_layers)
print(net.alpha_layers)
print(len(net.weights))
print(len(net.alpha_layers))
print(net.compute_cost(output, target))
print(net.backprop(target))
print(len(net.delta_layers))
# print(dnw)
# nabla_w = [np.zeros(w.shape) for w in net.weights]
# print(nabla_w)
# nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, dnw)]
# print(nabla_w)
# weights = net.weights
# n_weights = [w-(0.01)*nw
#                         for w, nw in zip(weights, nabla_w)]
# print("OLD")
# print(weights)
# print("NEW")
# print(n_weights)
