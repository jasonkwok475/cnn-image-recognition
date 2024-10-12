import numpy as np
import typing
from collections.abc import Generator

class ConvLayer2:
  padding = 0 #valid padding

  def __init__(self, kernel_size, output_channels, input_channels):
    self.kernel_size = kernel_size
    self.output_channels = output_channels
    self.input_channels = input_channels

    #Initializes a filter for each output_channel with size kernel_size, (9 -> xavier initialization)
    self.kernels = np.random.randn(output_channels, kernel_size, kernel_size, input_channels) / 9 

  #image is an n x n x input_channels array
  def getRegions(self, image) -> Generator:
    """
    Yields the m x m x n2 regions of the input image
    """
    h, w, _ = image.shape

    for i in range(h - self.padding * 2 - self.kernel_size):
      for j in range(w - self.padding * 2 - self.kernel_size):
        #Generates (n - 2)^2 m x m x n2 arrays of regions, where m = kernel size
        yield image[i:(i + self.kernel_size), j:(j + self.kernel_size)], i, j

  def propagate(self, input):
    """
    Generate an output for the convulutional layer 
    """
    h, w, _ = input.shape
    
    # Create an empty output array with zeros
    output = np.zeros((h - self.padding * 2 - self.kernel_size + 1, w - self.padding * 2 - self.kernel_size + 1, self.output_channels))

    for region, i, j in self.getRegions(input):

      #!double check axis, need to sum all elements of the 3d matrix to one number
      #this shouldn't be returning a matrix of the same number
      output[i, j] = np.sum(region * self.kernels, axis=(0,1,2,3)) 
      #!print(output[i, j])

    return output
  
  def backprop(self, gradient, learning_rate):
    return 0