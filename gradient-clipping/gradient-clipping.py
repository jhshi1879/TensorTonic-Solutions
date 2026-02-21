import numpy as np

def clip_gradients(g, max_norm):
  """
  Clip gradients using global norm clipping.
  """
  # Write code here
  g = np.asarray(g, dtype=float)

  l2_g = np.sqrt(np.sum(g**2))
  if l2_g >= max_norm > 0:
    g *= max_norm / l2_g
  return g