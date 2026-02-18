def linear_layer_forward(X, W, b):
  """
  Compute the forward pass of a linear (fully connected) layer.
  """
  # Write code here
  X = np.asarray(X, dtype=float) # (n, di)
  W = np.asarray(W, dtype=float) # (di, do)
  b = np.asarray(b, dtype=float) # (do, )

  n, d = X.shape

  return (X @ W + np.repeat(b[:,None], n, axis=1).T).tolist()