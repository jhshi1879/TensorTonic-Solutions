import numpy as np

def _entropy(y):
  """
  Helper: Compute Shannon entropy (base 2) for labels y.
  """
  y = np.asarray(y)
  if y.size == 0:
      return 0.0
  vals, counts = np.unique(y, return_counts=True)
  p = counts / counts.sum()
  p = p[p > 0]
  return float(-(p * np.log2(p)).sum()) if p.size else 0.0

def information_gain(y, split_mask):
  """
  Compute Information Gain of a binary split on labels y.
  Use the _entropy() helper above.
  """
  # Write code here
  y = np.asarray(y, dtype=int)
  yl = y[split_mask]
  yr = y[~split_mask]

  hl = _entropy(yl)
  hr = _entropy(yr)
  h = _entropy(y)

  nl = yl.size
  nr = yr.size
  n = y.size

  ig = h - (nl/n) * hl - (nr/n) * hr
  return ig
