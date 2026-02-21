import numpy as np

def adadelta_step(w, grad, E_grad_sq, E_update_sq, rho=0.9, eps=1e-6):
  """
  Perform one AdaDelta update step.
  """
  # Write code here
  g = np.asarray(grad, dtype=float)
  w = np.asarray(w, dtype=float)
  E_update_sq = np.asarray(E_update_sq, dtype=float)
  E_grad_sq = np.asarray(E_grad_sq, dtype=float)

  E_grad_sq = rho*E_grad_sq + (1 - rho) * g ** 2
  dw_t = - g * np.sqrt(E_update_sq + eps) / np.sqrt(E_grad_sq + eps)
  E_update_sq = rho * E_update_sq + (1 - rho) * dw_t ** 2
  w_t = w + dw_t
  return (w_t, E_grad_sq, E_update_sq)