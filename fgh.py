import numpy as np
from scipy.linalg import eigh
import time


def get_hamiltonian(grids, pot):
  """ Gets Hamiltonian for non-periodic potential (pot). 
  
  note(pedersor): slow. Could maybe speed up nested for loops with numba/jax?
  """

  num_grids = len(grids)
  ham = np.zeros([num_grids, num_grids])

  pi = np.pi

  spacing = grids[1] - grids[0]
  k = pi / spacing

  for i in range(num_grids):
    for j in range(i + 1):
      if i == j:
        ham[i, j] = 0.5 * k**2 / 3. + pot[i]
      else:
        ham[i, j] = k**2 / pi**2 * (-1.)**(j - i) / (j - i)**2
        # use Hermitian symmetry
        ham[j, i] = ham[i, j]

  return ham


def get_periodic_hamiltonian(pot):
  """ Gets Hamiltonian for periodic potential (pot). 
  
  note(pedersor): slow. Could maybe speed up nested for loops with numba/jax?
  """

  num_grids = len(grids)
  ham = np.zeros([num_grids, num_grids])
  pi = np.pi
  m = (num_grids - 1) / 2.

  for i in range(num_grids):
    for j in range(i + 1):
      if i == j:
        ham[i, j] = ((m * (m + 1)) / 3.) + pot[i]
        ham[i, j] = .5 * ((-1.)**(i - j)) * ham[i, j]
      else:
        ham[i, j] = (.5 * (np.cos(pi * (i - j) / (2. * m + 1.))) /
                     ((np.sin(pi * (i - j) / ((2. * m) + 1.)))**2))
        ham[i, j] = .5 * ((-1.)**(i - j)) * ham[i, j]
        # use Hermitian symmetry
        ham[j, i] = ham[i, j]

  return ham


def poschl_teller_potential(grids, lam):
  return -(lam * (lam + 1) / 2) / (np.cosh(grids))**2


def quartic_oscillator(grids, k):
  return .5 * k * (grids**4)


def kronig_penney_model(grids, a, b, v1, v2):
  """ assumes grids is a single unit cell and 0 <= grids <= a. 
  
  see https://en.wikipedia.org/wiki/Particle_in_a_one-dimensional_lattice
  """
  pot = np.where(grids < (a - b), v2, v1)
  return pot


if __name__ == "__main__":

  num_grid_points = 2000
  boundary = [-10, 10]

  # set of grids points, e.g. x \in [-10, 10]
  grids = np.linspace(boundary[0], boundary[1], num_grid_points)
  if 0 not in grids:
    # add extra grid point at origin to make symmetric
    grids = np.linspace(boundary[0], boundary[1], num_grid_points + 1)

  potential = poschl_teller_potential(grids, lam=2)
  t=time.time()
  H_sampled = get_hamiltonian(grids, potential)
  print("Time to get H: ", time.time()-t)
  

  eigvals, eigvecs = eigh(H_sampled)

  print('{:.20}'.format(eigvals[0]))
  print('{:.20}'.format(eigvals[1]))
  print('{:.20}'.format(eigvals[2]))
  print('{:.20}'.format(eigvals[3]))
