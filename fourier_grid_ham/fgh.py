import numpy as np


def get_open_ham(grids, pot):
  """ Gets Hamiltonian for non-periodic (open-boundary) potential (pot). """

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
        # Hermitian symmetry
        ham[j, i] = ham[i, j]

  return ham


def get_periodic_ham(grids, pot):
  """ Gets Hamiltonian for periodic potential (pot). """

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
        # Hermitian symmetry
        ham[j, i] = ham[i, j]

  return ham
