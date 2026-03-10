import numpy as np 

def jaccard_similarity(set_a, set_b):
  """
  Compute the Jaccard similarity between two item sets.
  """
  # Write code here
  a = np.asarray(set_a)
  b = np.asarray(set_b)

  a = np.unique(a)
  b = np.unique(b)

  eq = (a[:, None] == b[None, :])
  inter = eq.any(axis=0).sum()
  union = a.size + b.size - inter

  if union == 0:
    return 0

  return float(inter/union)
  