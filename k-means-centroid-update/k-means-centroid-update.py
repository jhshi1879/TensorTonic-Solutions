import numpy as np 

def k_means_centroid_update(points, assignments, k):
  """
  Compute new centroids as the mean of assigned points.
  """
  # Write code here
  p = np.asarray(points, dtype=float) 
  k_p = np.asarray(assignments, dtype=int) # (n_p, )

  if p.ndim == 1:
    p = p[:, None]

  n_p, d = p.shape

  # c = np.zeros((k, d))
  # for i in range(k):
  #   c[i] = np.mean(p[k_p==i], axis=0)

  k_p_one_hot = np.eye(k)[k_p] # (n_p, k)

  sums = k_p_one_hot.T @ p # (k d)
  counts = np.sum(k_p_one_hot, axis=0) # (k, )

  eps = 1e-15
  c = sums / (counts+eps)[:, None]

  return c.tolist()
  