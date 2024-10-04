from network import Network
from keras.datasets import mnist

(train_X, train_y), (test_X, test_y) = mnist.load_data()

network = Network(0.001, 10, 20)
print(network.propagate(train_X[0]))