import numpy as np

def k_means_assignment(points, centroids):
  """
  Assign each point to the nearest centroid.
  """
  p = np.asarray(points, dtype=float)
  c = np.asarray(centroids, dtype=float)

  if p.ndim == 1:
    p = p[:, None]
  if c.ndim == 1:
    c = c[:, None]
  
  dst = np.linalg.norm(p[:, None, :] - c[None, :, :], axis=-1) # (n_p, n_c)
  p_new = np.argmin(dst, axis=1) # (n_p, )
  return p_new.tolist()