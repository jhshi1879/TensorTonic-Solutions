import numpy as np 

def binning(values, num_bins, eps=1e-15):
  """
  Assign each value to an equal-width bin.
  """
  # Write code here
  x = np.asarray(values) # (n,)
  w = (np.max(x) - np.min(x)) / (num_bins + eps)
  # nodes = np.linspace(np.min(x), np.max(x), num_bins+1)
  bins = np.minimum(np.floor((x - np.min(x))/(w + eps)), np.ones_like(x) * (num_bins - 1))
  # bins = np.digitize(x, nodes[1:-1])
  return bins.tolist()