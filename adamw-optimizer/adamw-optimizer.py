import numpy as np

def adamw_step(w, m, v, grad, lr=0.001, beta1=0.9, beta2=0.999, weight_decay=0.01, eps=1e-8):
  """
  Perform one AdamW update step.
  """
  # Write code here
  m = np.asarray(m, dtype=float)
  w = np.asarray(w, dtype=float)
  v = np.asarray(v, dtype=float)
  g = np.asarray(grad, dtype=float)

  mt = beta1*m + (1-beta1)*g
  vt = beta2*v + (1-beta2)*np.square(g)
  wt = w - lr*(weight_decay*w) - lr*mt/(np.sqrt(vt) + eps)
  return (wt, mt, vt)