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
    
    self.last_input = input

    # Create an empty output array with zeros
    output = np.zeros((h - self.padding * 2 - self.kernel_size + 1, w - self.padding * 2 - self.kernel_size + 1, self.output_channels))

    for region, i, j in self.getRegions(input):

      #!double check axis, need to sum all elements of the 3d matrix to one number
      #this shouldn't be returning a matrix of the same number
      output[i, j] = np.sum(region * self.kernels, axis=(0,1,2,3)) 
      #!print(output[i, j])

    return output
  
  def getArrayRegions(self, image, shape):
    h, w, _ = image.shape
    h1, w2, _ = shape

    for i in range(h - self.padding * 2 - h1):
      for j in range(w - self.padding * 2 - w2):
        #Generates (n - 2)^2 m x m x n2 arrays of regions, where m = kernel size
        yield image[i:(i + h1), j:(j + w2)], i, j

  
  def backprop(self, dL_dout, learning_rate):

    dL_dkernel = np.zeros(self.kernels.shape)
    dL_dinput = np.zeros(self.last_input.shape)

    for region, i, j in self.getRegions(self.last_input):
      for f in range(self.kernel_size):
        dL_dkernel[f] += dL_dout[i, j, f] * region
      
    #!double check that this actually transforms the kernel correctly 
    #https://deeplearning.cs.cmu.edu/F21/document/recitation/Recitation5/CNN_Backprop_Recitation_5_F21.pdf

    filters = np.rot90(self.kernels, 2).reshape(10, 5, 5, 20) #! change this to reshape dynamically
    #filters = np.rot90(self.kernels, 2).swapaxes(0, 3) 

    num, _, _, _ = filters.shape
    pad = self.kernel_size - 1
    for k in range(num):
      for region_k, i_k, j_k, in self.getArrayRegions(np.pad(filters[k], [(pad, pad), (pad, pad), (0, 0)], mode='constant', constant_values=0), dL_dout.shape):
        dL_dinput[i_k, j_k, k] = np.sum(region_k * dL_dout, axis=(0,1,2)) 

    self.kernels -= learning_rate * dL_dkernel
    return dL_dinput