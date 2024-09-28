import numpy as np

class ConvLayer:
  def __init__(self, input_tensor, kernel_size, output_channels):
    self.input_tensor = input_tensor
    self.kernel_size = kernel_size
    self.output_channels = output_channels

    self.outputs = np.random.randn(output_channels, kernel_size, kernel_size) / 9 #double check what 9 does