#https://towardsdatascience.com/gentle-dive-into-math-behind-convolutional-neural-networks-79a07dd44cf9
import numpy as np
from conv import ConvLayer
from maxpool import MaxPool2
from fclayer import FCLayer

class Network:
  learning_rate = 0.001
  kernel_size = 5

  def __init__(self, learning_rate, conv1_outputs, conv2_outputs):
    #How fast the network will learn
    self.learning_rate = learning_rate
    
    #Number of ouput channels from each convulutional layer
    self.conv1_outputs = conv1_outputs
    self.conv2_outputs = conv2_outputs

    self.layers = Layers(
      ConvLayer(self.kernel_size, conv1_outputs),
      MaxPool2(),
      ConvLayer(self.kernel_size, conv2_outputs),
      MaxPool2(),
      FCLayer(20, conv2_outputs, "ReLU"),
      FCLayer(10, 20, "Softmax")
    )

  def propagate(self, input):
    output = self.layers.conv1.propagate(input)
    output = self.layers.pool1.propagate(output)
    output = self.layers.conv2.propagate(output)
    output = self.layers.pool2.propagate(output)
    output = self.layers.fclayer1.propagate(output.flatten())
    output = self.layers.fclayer2.propagate(output)

    return output


class Layers:
  def __init__(self, conv1, pool1, conv2, pool2, fc1, fc2):
    self.conv1 = conv1
    self.pool1 = pool1
    self.conv2 = conv2
    self.pool2 = pool2
    self.fclayer1 = fc1
    self.fclayer2 = fc2