#https://towardsdatascience.com/gentle-dive-into-math-behind-convolutional-neural-networks-79a07dd44cf9
import numpy as np
from structures.conv import ConvLayer

#!Combine all these into a general conv.py class in version 2
from structures.conv1 import ConvLayer1
from structures.conv2 import ConvLayer2

from structures.maxpool import MaxPool2
from structures.fclayer import FCLayer

#Ignore overflow warnings from numpy
np.seterr( over='ignore' )
#https://stackoverflow.com/questions/23128401/overflow-error-in-neural-networks-implementation

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
      FCLayer(20, conv2_outputs, "Sigmoid", True),
      FCLayer(10, 20, "Softmax", False)])
    
    print("MNIST CNN Initialized")

  def propagate(self, input, label):
    #add cross entropy here
    output =  self.layers.propagate(input)
    loss = -np.log(output[label])
    acc = 1 if np.argmax(output) == label else 0

    return output, loss, acc #probability outputs, cross entropy loss, accuracy
  
  def train(self, x_train, y_train):
    loss = 0
    correct = 0
    i = 0
    num, h, w = x_train.shape

    for i in range(num):
      output, l, acc = self.propagate(x_train[i], y_train[i])
      loss += l
      correct += acc

      self.layers.backprop(output, y_train[i], self.learning_rate)
      #label = y_train[i]

      if i % 100 == 99:
        print(
          '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
          (i + 1, loss / 100, correct))
        loss = 0
        correct = 0

      i += 1

    return

class Layers:
  def __init__(self, layers):
    self._layers = layers

  def propagate(self, input):
    output = input
    for layer in self._layers:
      output = layer.propagate(output)

    return output
  
  def backprop(self, output, label, learning_rate):
    
    #Initial gradients
    gradient = np.zeros(10)
    gradient[label] = -1 / output[label]
    i = 0
    for layer in reversed(self._layers):
      # if i > 0: return
      # i += 1
      #Layer receives dL/douput and returns dL/dinput
      gradient = layer.backprop(gradient, learning_rate)

    #Backprop is weight = weight - learning rate * output * error (from partial derivatives dL/dweight, where L is the cross entropy loss (output))
    return