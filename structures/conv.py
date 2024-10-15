import numpy as np
import typing
from collections.abc import Generator

class ConvLayer:
  padding = 0 #valid padding

  def __init__(self, kernel_size, output_channels, input_channels):
    self.kernel_size = kernel_size
    self.output_channels = output_channels
    self.input_channels = input_channels

    #Initializes a filter for each output_channel with size kernel_size, (9 -> xavier initialization)
    if input_channels == 1:
      self.kernels = np.random.randn(output_channels, kernel_size, kernel_size, input_channels) / 9 
    else:
      self.kernels = np.random.randn(output_channels, kernel_size, kernel_size) / 9 

  #image is an n x n array
  def getRegions(self, image) -> Generator:
    """
    Yields the n x n regions of the input image
    """
    h, w, outputs = image.shape[0], image.shape[1], image.shape[3:]

    for i in range(h - self.padding * 2 - self.kernel_size):
      for j in range(w - self.padding * 2 - self.kernel_size):
        #Generates (n - 2)^2 m x m arrays of regions, where m = kernel size
        yield image[i:(i + self.kernel_size), j:(j + self.kernel_size)], i, j

  def propagate(self, input):
    """
    Generate an output for the convulutional layer 
    """
    h, w, outputs = input.shape[0], input.shape[1], input.shape[3:]
    
    # Create an empty output array with zeros
    output = np.zeros((h - self.padding * 2, w - self.padding * 2, outputs if outputs else self.output_channels))

    for region, i, j in self.getRegions(input):
      #!double check axis here, should sum the whole matrix elements

      #!For now, just make two different files for conv1 and conv2
      output[i, j] = np.sum(region * self.kernels, axis=(1,2,3)) 

    return output