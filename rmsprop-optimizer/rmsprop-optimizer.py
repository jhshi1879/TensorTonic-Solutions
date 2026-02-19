import numpy as np

def rmsprop_step(w, g, s, lr=0.001, beta=0.9, eps=1e-8):
  """
  Perform one RMSProp update step.
  """
  # Write code here
  s = np.asarray(s, dtype=float)
  g = np.asarray(g, dtype=float)
  w = np.asarray(w, dtype=float)

  st = beta*s + (1-beta)*np.square(g)
  wt = w - lr * g / np.sqrt(st+eps)
  return (wt, st)