import math

def log_loss(y_true, y_pred, eps=1e-15):
  """
  Compute per-sample log loss.
  """
  # Write code here
  y = np.asarray(y_true, dtype=float)
  p = np.asarray(y_pred, dtype=float)

  p = np.clip(p, eps, 1-eps)
  l = - y * np.log(p) - (1 - y) * np.log(1-p)
  return l.tolist()