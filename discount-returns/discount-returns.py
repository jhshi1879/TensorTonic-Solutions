import numpy as np 

def discount_returns(rewards, gamma):
  """
  Compute the discounted return at every timestep.
  """
  r = np.asarray(rewards, dtype=float)
  gm = gamma

  T = len(rewards)

  g = np.empty_like(r)

  g[T-1] = r[T-1]

  for t in range(T-2,-1,-1):
    g[t] = r[t] + gm * g[t+1]

  return g.tolist()