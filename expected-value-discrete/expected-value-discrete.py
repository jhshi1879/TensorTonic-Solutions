import numpy as np

def expected_value_discrete(x, p):
  """
  Returns: float expected value
  """
  # Write code here
  x = np.asarray(x, dtype=float)
  p = np.asarray(p, dtype=float)
  if np.any(p < 0.) or np.any(p > 1.) or np.abs(np.sum(p) - 1.) > 1.0e-6:
    raise ValueError()
  return np.sum(x * p, axis=0)
