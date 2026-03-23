import numpy as np 

def value_iteration_step(values, transitions, rewards, gamma):
  """
  Perform one step of value iteration and return updated values.
  """
  # Write code here
  v = np.asarray(values, dtype=float)
  p = np.asarray(transitions, dtype=float)
  r = np.asarray(rewards, dtype=float)
  gm = gamma

  mod = r + gm * np.einsum(
      "abc,c->ab", p, v
  )
      
  v_new = np.max(
      mod, axis=-1
  )
  return v_new.tolist()