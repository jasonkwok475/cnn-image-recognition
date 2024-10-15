import numpy as np

class MaxPool2:

  def getRegions(self, image):
    """
    Yields the 2x2 pooling regions
    """
    h, w, _ = image.shape

    for i in range(h // 2):
      for j in range(w // 2) :
        yield image[(i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)], i, j

  def propagate(self, input):
    """
    Propagates the input forward through a MaxPool
    - input is a 3d numpy array
    """
    h, w, output_channels = input.shape
    output = np.zeros((h // 2, w // 2, output_channels))

    self.last_input = input

    for region, i, j in self.getRegions(input):
      output[i, j] = np.amax(region, axis=(0,1))

    return output

  def backprop(self, dL_dout, learning_rate):
    dL_dinput = np.zeros(self.last_input.shape)

    for im_region, i, j in self.getRegions(self.last_input):
      h, w, f = im_region.shape
      amax = np.amax(im_region, axis=(0, 1))

      for i2 in range(h):
        for j2 in range(w):
          for f2 in range(f):
            # If this pixel was the max value, copy the gradient to it.
            if im_region[i2, j2, f2] == amax[f2]:
              #!check if it should be = or += here
              dL_dinput[i * 2 + i2, j * 2 + j2, f2] += dL_dout[i, j, f2] 

    return dL_dinput