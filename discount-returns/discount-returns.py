import numpy as np 

def discount_returns(rewards, gamma):
  r = np.asarray(rewards, dtype=float)
  T = r.shape[0]

  if gamma == 0:
    return r.tolist()

  p = gamma ** np.arange(T)
  g = np.cumsum(r[::-1] / p) * p
  return g[::-1].tolist()