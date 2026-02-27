import numpy as np 

def min_max_scaling(data, eps=1e-15):
  """
  Scale each column of the data matrix to the [0, 1] range.
  """
  # Write code here
  x = np.asarray(data) # (n d)

  max_j = np.max(x, axis=0)
  min_j = np.min(x, axis=0)

  return ((x - min_j) / (max_j - min_j + eps)).tolist()