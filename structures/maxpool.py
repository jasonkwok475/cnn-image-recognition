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

    for region, i, j in self.getRegions(input):
      output[i, j] = np.amax(region, axis=(0,1))

    return output


