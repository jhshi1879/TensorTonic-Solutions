import numpy as np 

def precision_recall_at_k(recommended, relevant, k):
  """
  Compute precision@k and recall@k for a recommendation list.
  """
  # Write code here
  rec = np.asarray(recommended, dtype=int)
  rel = np.asarray(relevant, dtype=int)

  if rel.size == 0:
    return 0, 0

  vals, counts = np.unique(recommended[:k], return_counts=True)
  inter = 0
  for val in vals:
    if val in rel:
      inter += 1
  pk = inter/k
  rk = inter/rel.size
  return [pk, rk]