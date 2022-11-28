import time
from libc.math cimport sin, cos
import numpy as np
cimport numpy as np
from cython.parallel import prange


cpdef get_open_ham(double[:] grids, double[:] pot):
  """ Gets Hamiltonian for non-periodic (open-boundary) potential (pot). """

  cdef int num_grids = len(grids)
  cdef double spacing = grids[1] - grids[0]

  cdef np.ndarray[dtype=double, ndim=2] ham = np.zeros([num_grids, num_grids])
  
  cdef double pi = np.pi
  cdef double k = pi / spacing

  cdef int i, j
  for i in prange(num_grids, nogil=True):
    for j in range(i + 1):
      if i == j:
        ham[i, j] = 0.5 * k**2 / 3. + pot[i]
      else:
        ham[i, j] = k**2 / pi**2 * (-1.)**(j - i) / (j - i)**2
        # Hermitian symmetry
        ham[j, i] = ham[i, j]

  return ham

cpdef get_periodic_ham(double[:] grids, double[:] pot):
  """ Gets Hamiltonian for periodic potential (pot). """

  cdef int num_grids = len(grids)
  cdef double m = (num_grids - 1) / 2
  cdef double pi = np.pi

  cdef np.ndarray[dtype=double, ndim=2] ham = np.zeros([num_grids, num_grids])
  
  cdef int i, j
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
