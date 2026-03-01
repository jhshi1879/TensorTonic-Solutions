import numpy as np 

def detect_drift(reference_counts, production_counts, threshold):
  a = np.asarray(reference_counts, dtype=float)
  b = np.asarray(production_counts, dtype=float)

  p = a / a.sum()
  q = b / b.sum()

  tvd = 0.5 * np.sum(np.abs(p - q))

  return {
      "score": float(tvd),
      "drift_detected": bool(tvd > threshold)
  }