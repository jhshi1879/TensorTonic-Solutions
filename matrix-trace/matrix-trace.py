import numpy as np

def matrix_trace(A):
  """
  Compute the trace of a square matrix (sum of diagonal elements).
  """
  # Write code here
  A = np.asarray(A, dtype=float)
  n, _ = A.shape
  i = np.arange(n)
  tr_A = np.sum(A[i, i])
  return tr_A.tolist()
