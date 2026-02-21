import numpy as np

def dropout(x, p=0.5, rng=None):
  """
  Apply dropout to input x with probability p.
  Return (output, dropout_pattern).
  """
  # Write code here
  x = np.asarray(x, dtype=float)
  if rng:
    keep = rng.random(x.shape) < (1 - p)
  else:
    keep = np.random.random(x.shape) < (1 - p)

  prn = keep.astype(float) / (1 - p)
  return x*prn, prn