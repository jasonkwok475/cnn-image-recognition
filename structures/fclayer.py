import numpy as np

class FCLayer:

  def __init__(self, size, input_size, activiation_type, flatten):
    self.size = size
    self.input_size = size
    self.activation_type = activiation_type
    self.flatten = flatten #Whether to flatten the input or not

    #Dividing by input_size reduces the variance in initial weight values
    self.weights = ( 2 * np.random.rand(size, input_size * 16 if flatten == True else input_size) - 1 ) / input_size # weights initialized between -1 and 1
    self.biases = 2 * np.random.rand(size) - 1 # biases initialized between -1 and 1 

  def propagate(self, input):
    """
    Propagates input through a fully-connected layer
    - input is a n x 1 array
    """
    #might need to save output here to back propagate
    if self.flatten == True:
      output = np.dot(self.weights, input.flatten()) + self.biases
    else:
      output = np.dot(self.weights, input) + self.biases

    match self.activation_type:
      case "ReLU":
        return np.maximum(output, 0)
      case "Softmax":
        exp = np.exp(output)
        return exp / np.sum(exp, axis=0) #double check this formula
     

    #matrix multiplication to do weights * x + biases and then sum down to a 1 x n vector array for the output

