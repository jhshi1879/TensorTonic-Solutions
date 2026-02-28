import numpy as np 

def binning(values, num_bins, eps=1e-15):
  """
  Assign each value to an equal-width bin.
  """
  # Write code here
  x = np.asarray(values) # (n,)
  w = (np.max(x) - np.min(x)) / (num_bins + eps)
  bins = np.minimum(np.floor((x - np.min(x))/(w + eps)), num_bins - 1)
  return bins.tolist()