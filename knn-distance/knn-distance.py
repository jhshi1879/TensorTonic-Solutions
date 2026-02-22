import numpy as np

def knn_distance(X_train, X_test, k):
  """
  Compute pairwise distances and return k nearest neighbor indices.
  """
  # Write code here
  X_train = np.asarray(X_train, dtype=float)
  X_test = np.asarray(X_test, dtype=float)

  if X_train.ndim == 1:
    X_train = X_train.reshape(-1, 1)

  if X_test.ndim == 1:
    X_test = X_test.reshape(-1, 1)

  n_train, d = X_train.shape 
  n_test, d = X_test.shape
  
  dst = np.linalg.norm(X_train[None, :, :] - X_test[:, None, :], axis=-1) # (n_test, n_train)
  idx = np.argsort(dst, axis=1)
  top_k = idx[:, :k]

  if k > n_train:
    padding = np.ones((n_test, k-n_train), dtype=int) * (-1)
    top_k = np.hstack([top_k, padding])

  return top_k