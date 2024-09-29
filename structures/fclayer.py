import numpy as np

class FCLayer:

  def __init__(self, size, input_size, activiation_type):
    self.size = size
    self.input_size = size
    self.activation_type = activiation_type

    self.weights = 2 * np.random.rand(size, input_size) - 1 # weights initialized between -1 and 1
    self.biases = 2 * np.random.rand(size, input_size) - 1 # biases initialized between -1 and 1 

  def propagate(self, input):
    """
    Propagates input through a fully-connected layer
    - input is a n x 1 array
    """
    #might need to save output here to back propagate
    output = np.dot(self.weights * input) + self.biases

    match self.activation_type:
      case "ReLU":
        return np.max(output, 0)
      case "Softmax":
        exp = np.exp(output)
        return exp / np.sum(exp, axis=0) #double check this formula
     

    #matrix multiplication to do weights * x + biases and then sum down to a 1 x n vector array for the output

