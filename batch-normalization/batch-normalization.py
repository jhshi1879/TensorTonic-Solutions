import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5):
  """
  Forward-only BatchNorm for (N,D) or (N,C,H,W).
  """
  # Write code here
  x = np.asarray(x, dtype=float)
  if x.ndim not in (2,4):
    raise ValueError("x must be 2D or 4D")

  gamma = np.asarray(gamma, dtype=float)
  beta = np.asarray(beta, dtype=float) # (C, )

  broadcast_shape_ = (1, -1) if x.ndim == 2 else (1, -1, 1, 1)
  gamma = gamma.reshape(broadcast_shape_)
  beta = beta.reshape(broadcast_shape_)

  reduction_axis_ = 0 if x.ndim == 2 else (0,2,3)
  mu = np.mean(x, axis=reduction_axis_, keepdims=True)
  sigma_sq = np.mean(np.square(x - mu), axis=reduction_axis_, keepdims=True)
  
  x_hat = (x - mu) / np.sqrt(sigma_sq + eps)
  y = gamma * x_hat + beta
  return y
