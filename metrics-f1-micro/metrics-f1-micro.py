import numpy as np 

def f1_micro(y_true, y_pred) -> float:
  """
  Compute micro-averaged F1 for multi-class integer labels.
  """
  # Write code here
  y_t = np.asarray(y_true, dtype=int) # (n, )
  y_p = np.asarray(y_pred, dtype=int) # (n, )

  mathces = (y_t == y_p)
  tp = np.sum(mathces)
  fp = fn = np.sum(~mathces)
  f1 = (2*tp) / (2*tp+fp+fn)
  return f1