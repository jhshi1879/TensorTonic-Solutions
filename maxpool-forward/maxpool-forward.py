import numpy as np 

def maxpool_forward(X, pool_size, stride):
  """
  Compute the forward pass of 2D max pooling.
  """
  # Write code here
  s = stride
  p = pool_size

  X = np.asarray(X, dtype=float)
  h, w = X.shape

  h_out, w_out = (h-p)//s + 1, (w-p)//s + 1

  a = np.arange(p)[None, :, None, None]
  b = np.arange(p)[None, None, None, :]
  i_s = np.arange(0, h_out*s, s)[:, None, None, None]
  j_s = np.arange(0, w_out*s, s)[None, None, :, None]

  # X[i*s+a, j*s+b] # (h_out, p, w_out, p)
  out = np.max(X[i_s+a, j_s+b], axis=(1,3))
  return out.tolist()