import time
import numpy as np
from test_cython import get_open_ham
from scipy.linalg import eigh

def poschl_teller_potential(grids, lam):
  return -(lam * (lam + 1) / 2) / (np.cosh(grids))**2

if __name__ == "__main__":

  num_grid_points = 8000
  boundary = [-10, 10]

  # set of grids points, e.g. x \in [-10, 10]
  grids = np.linspace(boundary[0], boundary[1], num_grid_points)
  if 0 not in grids:
    # add extra grid point at origin to make symmetric
    grids = np.linspace(boundary[0], boundary[1], num_grid_points + 1)

  potential = poschl_teller_potential(grids, lam=2)
  t=time.time()
  H_sampled = get_open_ham(grids, potential)
  print("Time to get H: ", time.time()-t)

  eigvals, eigvecs = eigh(H_sampled)
  print('{:.20}'.format(eigvals[0]))
  print('{:.20}'.format(eigvals[1]))
  print('{:.20}'.format(eigvals[2]))
  print('{:.20}'.format(eigvals[3]))
