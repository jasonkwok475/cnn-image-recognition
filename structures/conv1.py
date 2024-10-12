import numpy as np
import typing
from collections.abc import Generator

class ConvLayer1:
  padding = 0 #valid padding

  def __init__(self, kernel_size, output_channels):
    self.kernel_size = kernel_size
    self.output_channels = output_channels

    #Initializes a filter for each output_channel with size kernel_size, (9 -> xavier initialization)
    self.kernels = np.random.randn(output_channels, kernel_size, kernel_size) / 9 

  #image is an n x n array
  def getRegions(self, image) -> Generator:
    """
    Yields the n x n regions of the input image
    """
    h, w = image.shape

    for i in range(h - self.padding * 2 - self.kernel_size):
      for j in range(w - self.padding * 2 - self.kernel_size):
        #Generates (n - 2)^2 m x m arrays of regions, where m = kernel size
        yield image[i:(i + self.kernel_size), j:(j + self.kernel_size)], i, j

  def propagate(self, input):
    """
    Generate an output for the convulutional layer 
    """
    h, w = input.shape
    
    # Create an empty output array with zeros
    output = np.zeros((h - self.padding * 2 - self.kernel_size + 1, w - self.padding * 2 - self.kernel_size + 1, self.output_channels))

    for region, i, j in self.getRegions(input):
      output[i, j] = np.sum(region * self.kernels, axis=(1,2)) 

    return output

  def backprop(self, gradient, learning_rate):
    return 0