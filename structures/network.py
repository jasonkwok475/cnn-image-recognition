#https://towardsdatascience.com/gentle-dive-into-math-behind-convolutional-neural-networks-79a07dd44cf9
import numpy as np
from conv import ConvLayer

#!Combine all these into a general conv.py class
from conv1 import ConvLayer1
from conv2 import ConvLayer2

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

    self.layers = Layers([
      ConvLayer1(self.kernel_size, conv1_outputs),
      MaxPool2(),
      ConvLayer2(self.kernel_size, conv2_outputs, conv1_outputs),
      MaxPool2(),
      FCLayer(20, conv2_outputs, "ReLU", True),
      FCLayer(10, 20, "Softmax", False)])
    
    print("MNIST CNN Initialized")

  def propagate(self, input):
    return self.layers.propagate(input)


class Layers:
  def __init__(self, layers):
    self._layers = layers

  def propagate(self, input):
    output = input
    for layer in self._layers:
      output = layer.propagate(output)

    return output