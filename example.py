import time
import numpy as np
import fourier_grid_ham.fgh_fast as fgh
import fourier_grid_ham.fgh as fgh_slow
from scipy.linalg import eigh


def poschl_teller_potential(grids, lam):
  return -(lam * (lam + 1) / 2) / (np.cosh(grids))**2


def quartic_oscillator(grids, k):
  return .5 * k * (grids**4)


def kronig_penney(grids, a, b, v0):
  """Kronig-Penney model potential. For more information, see:

    https://en.wikipedia.org/wiki/Particle_in_a_one-dimensional_lattice#Kronig%E2%80%93Penney_model

    Args:
      grids: numpy array of grid points for evaluating 1d potential.
        (num_grids,)
      a: periodicity of 1d lattice
      b: width of potential well
      v0: negative float. It is the depth of potential well.

    Returns:
      vp: Potential on grid.
        (num_grid,)
    """
  if v0 >= 0:
    raise ValueError('v0 is expected to be negative but got %4.2f.' % v0)
  if b >= a:
    raise ValueError('b is expected to be less than a but got %4.2f.' % b)

  pot = np.where(grids < (a - b), 0., v0)

  return pot


if __name__ == "__main__":

  # Kronig-Penney model (periodic) example
  boundary = [0, 3]
  num_grid_points = 10
  grids = np.linspace(*boundary, num_grid_points, endpoint=False)
  # Note(pedersor): use endpoint=False for periodic grids.

  potential = kronig_penney(grids, a=3, b=0.5, v0=-1)

  t = time.time()
  ham = fgh.get_periodic_ham(grids, potential)
  print(ham)
  print("Time to get H: ", time.time() - t)

  t = time.time()
  ham = fgh_slow.get_periodic_ham(grids, potential)
  print(ham)
  print("Time to get H: ", time.time() - t)

  t = time.time()
  eigvals, eigvecs = eigh(ham)
  print("Time to solve eigh(H): ", time.time() - t)

  print("Lowest eigenvalues: ")
  print('e_0 = ' + '{:.20}'.format(eigvals[0]))
  print('e_1 = ' + '{:.20}'.format(eigvals[1]))
  print('e_2 = ' + '{:.20}'.format(eigvals[2]))
  print('e_3 = ' + '{:.20}'.format(eigvals[3]))

  # reference values
  # e_0 = 0.1902143714490897

  num_grid_points = 8000
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
