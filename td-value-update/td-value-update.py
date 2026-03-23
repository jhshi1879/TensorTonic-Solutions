import numpy as np

def td_value_update(V, s, r, s_next, alpha, gamma):
  """
  Returns: updated value function V_new
  """
  # Write code here
  v = np.asarray(V, dtype=float)
  s = np.asarray(s, dtype=int)
  r = np.asarray(r, dtype=float)
  s_nx = np.asarray(s_next, dtype=int)
  gm = gamma
  al = alpha

  dq = r + gm * v[s_nx] - v[s]
  new_v = v.copy()
  np.add.at(new_v, s, al*dq)

  return new_v