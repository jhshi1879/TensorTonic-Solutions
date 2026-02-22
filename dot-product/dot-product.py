import numpy as np
"""
What are the dtypes of x and y
What are the return dtypes
"""
def dot_product(x, y):
  """
  Compute the dot product of two 1D arrays x and y.
  Must return a float.
  """
  # Write code here
  x = np.asarray(x, dtype=float)
  y = np.asarray(y, dtype=float)
  return np.sum(x*y, axis=0)