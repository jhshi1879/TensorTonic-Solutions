def sarsa_update(q_table, state, action, reward, next_state, next_action, alpha, gamma):
  """
  Perform one SARSA update and return the updated Q-table.
  """
  # Write code here
  q = np.asarray(q_table, dtype=float)
  s = np.asarray(state, dtype=int)
  a = np.asarray(action, dtype=int)
  r = np.asarray(reward, dtype=float)
  s_nx = np.asarray(next_state, dtype=int)
  a_nx = np.asarray(next_action, dtype=int)
  al = alpha
  gm = gamma

  dq = r + gm * q[s_nx, a_nx] - q[s, a]
  
  q_nx = q.copy()

  np.add.at(q_nx, (s, a), al * dq)

  return q_nx