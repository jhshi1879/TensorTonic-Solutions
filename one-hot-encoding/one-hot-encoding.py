import numpy as np

def one_hot(y, num_classes=None):
  """
  Convert integer labels y ∈ {0,...,K-1} into one-hot matrix of shape (N, K).
  """
  # Write code here
  y = np.asarray(y, dtype=int)
  if num_classes:
    k = num_classes
  else:
    k = np.max(y) + 1
  y_one_hot = np.eye(k)[y]
  return y_one_hot