import numpy as np

def nadam_step(w, m, v, grad, lr=0.002, beta1=0.9, beta2=0.999, eps=1e-8):
  """
  Perform one Nadam update step.
  """
  # Write code here
  w = np.asarray(w, dtype=float)
  m = np.asarray(m, dtype=float)
  v = np.asarray(v, dtype=float)
  gt = np.asarray(grad, dtype=float)

  mt = beta1 * m + (1 - beta1) * gt
  vt = beta2 * v + (1 - beta2) * gt ** 2
  wt = w - lr * (beta1*mt + (1-beta1)*gt) / (np.sqrt(vt) + eps)
  return (wt, mt, vt)