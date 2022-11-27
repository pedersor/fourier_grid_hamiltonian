import time
import numpy as np
cimport numpy as np
from cython.parallel import prange

def test_fun(int maxval):

  cdef unsigned long long int total
  cdef int k
  cdef float t1, t2, t

  t1=time.time()

  for k in range(maxval):
      total = total + k

  t2=time.time()
  t = t2-t1
  print("%.100f" % t)

  return t

cpdef get_open_ham(double[:] grids, double[:] pot):

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
        # use Hermitian symmetry
        ham[j, i] = ham[i, j]

  return ham
