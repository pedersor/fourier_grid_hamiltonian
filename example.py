import time
import numpy as np
import fourier_grid_ham.fgh_fast as fgh
from scipy.linalg import eigh


def poschl_teller_potential(grids, lam):
  return -(lam * (lam + 1) / 2) / (np.cosh(grids))**2


def quartic_oscillator(grids, k):
  return .5 * k * (grids**4)


if __name__ == "__main__":

  num_grid_points = 4000
  boundary = [-10, 10]

  # set of grids points, e.g. x \in [-10, 10]
  grids = np.linspace(boundary[0], boundary[1], num_grid_points)
  if 0 not in grids:
    # add extra grid point at origin to make symmetric
    grids = np.linspace(boundary[0], boundary[1], num_grid_points + 1)

  potential = poschl_teller_potential(grids, lam=2)
  t = time.time()
  ham = fgh.get_open_ham(grids, potential)
  print("Time to get H: ", time.time() - t)

  t = time.time()
  eigvals, eigvecs = eigh(ham)
  print("Time to solve eigh(H): ", time.time() - t)

  print("Lowest eigenvalues: ")
  print('e_0 = ' + '{:.20}'.format(eigvals[0]))
  print('e_1 = ' + '{:.20}'.format(eigvals[1]))
  print('e_2 = ' + '{:.20}'.format(eigvals[2]))
  print('e_3 = ' + '{:.20}'.format(eigvals[3]))
