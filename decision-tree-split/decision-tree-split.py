import numpy as np

def decision_tree_split(X, y):
  X = np.asarray(X, dtype=float) # (n, d)
  y = np.asarray(y, dtype=int) # (n, )
  n, d = X.shape

  classes, y_enc = np.unique(y, return_inverse=True) # classes: (K, ) y_enc: (n, )
  K = len(classes)

  # sort each feature
  order = np.argsort(X, axis=0) # (n, d)
  Xs = np.take_along_axis(X, order, axis=0) # (n, d)
  ys = y_enc[order]  # (n d)
  
  Y = np.eye(K)[ys] # (n d K)

  csum = np.cumsum(Y, axis=0) # (n d K)
  total = csum[-1, :, :]# (d K)
  
  valid = Xs[:-1] != Xs[1:] # (n-1 d)

  left_count = csum[:-1, :, :] # (n-1 d k)
  right_count = total - left_count # (n-1 d k)

  n_left = np.arange(1, n)[:, None] # (n-1 1)
  n_right = n - n_left # (n-1 1)

  pl2 = np.square(left_count/n_left[:, :, None]) # (n-1 d)
  gl = 1 - pl2.sum(axis=2) # (n-1 d)

  pr2 = np.square(right_count/n_right[:, :, None]) # (n-1 d)
  gr = 1 - pr2.sum(axis=2) # (n-1 d)

  g_split = (n_left/n) * gl + (n_right/n) * gr # (n-1 d)
  g_split[~valid] = np.inf

  parent_counts = total[0] # (K, )
  p_pa2 = np.square(parent_counts / n) # (K, )
  g_pa = 1. - p_pa2.sum()

  g_gain = g_pa - g_split # (n-1 d)
  idx = np.argmax(g_gain.ravel(order="C")) # (n-1*d) -> int
  i, f = np.unravel_index(idx, g_gain.shape, order="C") # (n-1, d)
  
  threshold = (Xs[i, f] + Xs[i+1, f]) / 2.
  return [f.tolist(), threshold.tolist()]