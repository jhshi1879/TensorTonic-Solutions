import numpy as np 

def precision_recall_at_k(recommended, relevant, k):
  """
  Compute precision@k and recall@k for a recommendation list.
  """
  # Write code here
  rec = np.asarray(recommended, dtype=int)
  rel = np.asarray(relevant, dtype=int)
  top_k = rec[:k]

  if len(rel) == 0:
    return 0, 0

  vals, counts = np.unique(recommended[:k], return_counts=True)
  inter = np.sum(np.isin(top_k, rel))
  pk = inter/k
  rk = inter/len(rel)
  return [pk, rk]