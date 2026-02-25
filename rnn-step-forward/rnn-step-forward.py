import numpy as np

def rnn_step_forward(x_t, h_prev, Wx, Wh, b):
  """
  Returns: h_t of shape (H,)
  """
  # Write code here
  x_t = np.asarray(x_t) # (d, )
  h_prev = np.asarray(h_prev) # (h,)
  Wx = np.asarray(Wx) # (d, h)
  Wh = np.asarray(Wh) # (h, h)
  b = np.asarray(b) # (h, )

  ht = np.tanh(x_t@Wx + h_prev@Wh + b)
  return ht