import numpy as np

def apply_homogeneous_transform(T, points):
  """
  Apply 4x4 homogeneous transform T to 3D point(s).
  """
  # Your code here
  p = np.asarray(points, dtype=float) # (n 3) (, 3)
  if p.ndim == 1:
    p = p[None, :] # (n 3)  n = 1, ...
  n, d = p.shape
  ph = np.ones((n,d+1))
  ph[:, :3] = p 
  Tr = np.asarray(T, dtype=float) # (4 4)
  return (Tr @ ph.T)[:3].transpose(1,0).squeeze().tolist()