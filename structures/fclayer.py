import numpy as np

class FCLayer:

  def __init__(self, size, input_size, activiation_type, flatten):
    self.size = size
    self.input_size = size
    self.activation_type = activiation_type
    self.flatten = flatten #Whether to flatten the input or not

    #Dividing by input_size reduces the variance in initial weight values
    #!Change the *16 portion to be dependant on a class initialization parameter

    inputSize = (input_size * 16) if flatten == True else input_size

    self.weights = ( 2 * np.random.rand(size, inputSize) - 1 ) / inputSize # weights initialized between -1 / n and 1 / n
    self.biases = 2 * np.random.rand(size) - 1 # biases initialized between -1 and 1 

  def propagate(self, input):
    """
    Propagates input through a fully-connected layer
    - input is a n x 1 array
    """
    _input = input.flatten() if self.flatten else input
    output = np.dot(self.weights, _input) + self.biases

    #save forward values for back propagation
    self.last_input = _input #flattened input, if applicable
    self.last_input_shape = input.shape #input shape before flattening
    self.last_totals = output

    match self.activation_type:
      case "ReLU":
        return np.maximum(output, 0)
      case "Softmax":
        exp = np.exp(output)
        return exp / np.sum(exp, axis=0) #double check this formula
      
  def backprop(self, d_l_d_out, learning_rate):
    match self.activation_type:
      case "ReLU":
        for i, gradient in enumerate(d_l_d_out):
          if gradient == 0:
            continue
          
          d_out_d_t = np.ones(self.size)

          # Gradients of totals against weights/biases/input
          d_t_d_w = self.last_input
          d_t_d_b = 1
          d_t_d_inputs = self.weights

          # Gradients of loss against totals
          d_L_d_t = gradient * d_out_d_t

          # Gradients of loss against weights/biases/input
          d_L_d_w = d_t_d_w[np.newaxis].T @ d_L_d_t[np.newaxis] #! make sure to understand what is going on here with dimensions
          d_L_d_b = d_L_d_t * d_t_d_b
          d_L_d_inputs = d_t_d_inputs.T @ d_L_d_t #!double check all array dimensions, esp .T here

          # Update weights / biases
          print(d_L_d_w.T) #!for some reason both W and B here all have the same number for each element, might be due to summing axis or smth
          print(d_L_d_b)
          self.weights -= learning_rate * d_L_d_w.T
          self.biases -= learning_rate * d_L_d_b
          return d_L_d_inputs.reshape(self.last_input_shape)
      case "Softmax":

        #!copied and pasted, make sure you understand what this is and modify it for own needs
        # We know only 1 element of d_L_d_out will be nonzero
        for i, gradient in enumerate(d_l_d_out):
          if gradient == 0:
            continue

          # e^totals
          t_exp = np.exp(self.last_totals)

          # Sum of all e^totals
          S = np.sum(t_exp)

          # Gradients of out[i] against totals
          d_out_d_t = -t_exp[i] * t_exp / (S ** 2)
          d_out_d_t[i] = t_exp[i] * (S - t_exp[i]) / (S ** 2)

          # Gradients of totals against weights/biases/input
          d_t_d_w = self.last_input
          d_t_d_b = 1
          d_t_d_inputs = self.weights

          # Gradients of loss against totals
          d_L_d_t = gradient * d_out_d_t

          # Gradients of loss against weights/biases/input
          d_L_d_w = d_t_d_w[np.newaxis].T @ d_L_d_t[np.newaxis]
          d_L_d_b = d_L_d_t * d_t_d_b
          d_L_d_inputs = d_t_d_inputs.T @ d_L_d_t #!double check all array dimensions, esp .T here

          # Update weights / biases
          self.weights -= learning_rate * d_L_d_w.T
          self.biases -= learning_rate * d_L_d_b
          return d_L_d_inputs.reshape(self.last_input_shape)