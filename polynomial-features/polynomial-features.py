import numpy as np 

def polynomial_features(values, degree):
  """
  Generate polynomial features for each value up to the given degree.
  """
  # Write code here
  x = np.asarray(values)
  d = degree

  out = []
  for i in range(0, d+1):
    out += [x ** i]
  return np.stack(out, axis=1).tolist()
  