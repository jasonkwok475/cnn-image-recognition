#https://towardsdatascience.com/gentle-dive-into-math-behind-convolutional-neural-networks-79a07dd44cf9
from conv import ConvLayer
from perceptron import Perceptron

class Network:
  learning_rate = 0.001

  def init(self, learning_rate, conv1_outputs, conv2_outputs):
    #How fast the network will learn
    self.learning_rate = learning_rate
    
    #Number of output channels from each convolutional layer
    self.conv1_outputs = conv1_outputs
    self.conv2_outputs = conv2_outputs

