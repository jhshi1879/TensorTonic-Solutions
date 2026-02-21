import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5):
  """
  Forward-only BatchNorm for (N,D) or (N,C,H,W).
  """
  # Write code here
  x = np.asarray(x, dtype=float)
  gamma = np.asarray(gamma, dtype=float)
  beta = np.asarray(beta, dtype=float)

  if x.ndim == 2:
    mu = np.mean(x, axis=0, keepdims=True)
    sigma_sq = np.mean(np.square(x - mu), axis=0, keepdims=True)
  elif x.ndim == 4:
    mu = np.mean(x, axis=(0,2,3), keepdims=True)
    sigma_sq = np.mean(np.square(x - mu), axis=(0,2,3), keepdims=True)
  else:
    raise Exception()
  
  x_hat = (x - mu) / np.sqrt(sigma_sq + eps)
  if x.ndim == 2:
    gamma = gamma[None, :]
    beta = beta[None, :]
  elif x.ndim == 4:
    gamma = gamma[None, :, None, None]
    beta = beta[None, :, None, None]
  y = gamma * x_hat + beta
  return y
