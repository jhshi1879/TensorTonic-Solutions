import numpy as np

def conv2d(x, W, b):
  """
  Simple 2D convolution layer forward pass.
  Valid padding, stride=1.
  """
  # Write code here
  x = np.asarray(x, dtype=float)
  W = np.asarray(W, dtype=float)
  b = np.asarray(b, dtype=float)

  n, c_in, h, w = x.shape
  c_out, _, kh, kw = W.shape

  h_out, w_out = (h - kh + 1), (w - kw + 1)

  i = np.arange(h_out)[:, None, None, None]
  j = np.arange(w_out)[None, None, :, None]

  u = np.arange(kh)[None, :, None, None]
  v = np.arange(kw)[None, None, None, :]

  patches = x[:, :, i+u, j+v] # n, c_in, h_out, kh, w_out, wh
  x_W = np.einsum("abcdef,gbdf->aceg", patches, W).transpose(0,3,1,2) # (n h w c_out)

  y = x_W + b[None,:,None,None]
  return y
  
  

    