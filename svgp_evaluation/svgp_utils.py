import numpy as np


def fdiff_jac(f,x0,h=1e-6):
  """Compute the jacobian of f with 
  central difference
  """
  h2   = h/2.0
  dim  = len(x0)
  Ep   = x0 + h2*np.eye(dim)
  Fp   = np.array([f(e) for e in Ep])
  Em   = x0 - h2*np.eye(dim)
  Fm   = np.array([f(e) for e in Em])
  jac = (Fp - Fm)/(h)
  return jac.T

def meshgrid_uniform(xlo, xhi, ylo, yhi, nx, ny):
  xx, yy = np.meshgrid(np.linspace(xlo, xhi, nx),
                  np.linspace(ylo, yhi, ny))
  return np.vstack([xx.reshape(1,-1), yy.reshape(1,-1)]) 