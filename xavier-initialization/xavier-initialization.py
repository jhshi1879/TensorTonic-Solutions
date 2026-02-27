import numpy as np

def xavier_initialization(W, fan_in, fan_out):
  """
  Scale raw weights to Xavier uniform initialization.
  """
  # Write code here
  l = np.sqrt(6 / (fan_in + fan_out))
  W = np.asarray(W)
  W = (W * 2 * l) - l
  return W.tolist()