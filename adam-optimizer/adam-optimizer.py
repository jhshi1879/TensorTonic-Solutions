import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
  """
  One Adam optimizer update step.
  Return (param_new, m_new, v_new).
  """
  # Write code here
  param = np.asarray(param, dtype=float)
  gt = np.asarray(grad, dtype=float)
  m =  np.asarray(m, dtype=float)
  v = np.asarray(v, dtype=float)
  t = float(t)

  mt = beta1 * m + (1. - beta1) * gt
  vt = beta2 * v + (1. - beta2) * gt**2

  mt_hat = mt / (1. - np.power(beta1, t))
  vt_hat = vt / (1. - np.power(beta2, t))
  param -= lr * mt_hat / (np.sqrt(vt_hat)+eps)
  return (param, mt, vt)