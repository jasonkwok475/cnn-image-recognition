import numpy as np
import typing
from collections.abc import Generator

class ConvLayer:
  padding = 1 #valid padding

  def __init__(self, input_tensor, kernel_size, output_channels):
    self.input_tensor = input_tensor
    self.kernel_size = kernel_size
    self.output_channels = output_channels

    self.kernel = np.random.randn(output_channels, kernel_size, kernel_size) / 9 #double check what 9 does, xavier initialization?

  #image is an n x n array
  def getRegions(self, image) -> Generator:
    """
    Yields the n x n regions of the input image
    """
    h, w = image.shape

    for i in range(h - self.padding*2):
      for j in range(w - self.padding*2):
        #Generates n - 2 m x m arrays of regions, where m = kernel size
        yield image[i:(i + self.kernel_size), j:(j + self.kernel_size)], i, j

  def propagate(self, input):
    h, w = input.shape
    
    # Create an empty output array with zeros
    output = np.zeros((h - self.padding*2, w - self.padding*2, self.output_channels))

    for region, i, j in self.getRegions(input):
      output[i, j] = np.sum(region * self.kernel) #not finished here

    return output

    