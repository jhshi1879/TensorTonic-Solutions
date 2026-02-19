import numpy as np

def adagrad_step(w, g, G, lr=0.01, eps=1e-8):
  """
  Perform one AdaGrad update step.
  """
  # Write code here
  G = np.asarray(G, dtype=float)
  g = np.asarray(g, dtype=float)
  w = np.asarray(w, dtype=float)
  
  Gt = G + np.square(g)
  wt = w - lr * g / np.sqrt(Gt+eps)
  return (wt, Gt)