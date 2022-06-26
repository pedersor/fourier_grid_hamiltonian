from itertools import cycle

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.integrate import simps

from numpy.polynomial.polynomial import polyfit
from scipy import stats
import sys

pi = np.pi


def rsquared(x, y):
  """ Return R^2 where x and y are array-like."""
  slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
  return r_value**2


def H(L, V):
  N = len(V)
  Hij = np.zeros([N, N])
  pi = np.pi
  k = pi / (L / N)

  for i in range(N):
    for j in range(i + 1):
      if i == j:
        Hij[i, j] = 0.5 * k**2 / 3. + V[i]
      else:
        Hij[i, j] = k**2 / pi**2 * (-1.)**(j - i) / (j - i)**2
        Hij[j, i] = Hij[i, j]  # use Hermitian symmetry

  return Hij


def H_per(V):
  N = len(V)
  Hij = np.zeros([N, N])
  pi = np.pi
  M = (N - 1) / 2.

  for i in range(N):
    for j in range(i + 1):
      if i == j:
        Hij[i, j] = ((M * (M + 1)) / 3.) + V[i]
        Hij[i, j] = .5 * ((-1.)**(i - j)) * Hij[i, j]
      else:
        Hij[i,
            j] = .5 * (np.cos(pi * (i - j) /
                              (2. * M + 1.))) / ((np.sin(pi * (i - j) /
                                                         ((2. * M) + 1.)))**2)
        Hij[i, j] = .5 * ((-1.)**(i - j)) * Hij[i, j]
        Hij[j, i] = Hij[i, j]  # use Hermitian symmetry

  return Hij


def V(x):
  # return  (x ** 4 - 20 * x ** 2)
  #return (.5 * x**2)
  return -(3 / 8) / ((np.cosh(x))**2)


def quartic(x):
  return .5 * (x**4)


def V_per(x):
  pi = np.pi
  a = 2. * pi
  b = 1.
  v0 = 1.5 / (2. * pi)
  if x < (a - b):
    return 0.
  else:
    return -1. * v0


params = {'mathtext.default': 'regular'}
plt.rcParams.update(params)
plt.rcParams['axes.axisbelow'] = True
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 9
fig_size[1] = 6
plt.rcParams["figure.figsize"] = fig_size
fig, ax = plt.subplots()

# open boundary w.f. -----------------------
N = 8500
L = 20
x_vals = np.linspace(-L / 2, +L / 2, N, endpoint=False)

V_sampled = [quartic(x) for x in x_vals]
H_sampled = H(L, V_sampled)

E, psi = eigh(H_sampled)

print('{:.20}'.format(E[0]))
print('{:.20}'.format(E[1]))
print('{:.20}'.format(E[2]))
print('{:.20}'.format(E[3]))
sys.exit()

N_plot_min = 0  # quantum number of first eigenfunction to plot
N_plot = 3  # number of eigenfunctions to plot

WF_scale_factor = (np.max(V_sampled) - np.min(V_sampled)) / N_plot
plt.plot(x_vals, V_sampled, ls="-", c="k", lw=2, label="$V(x)$")

for i in range(N_plot_min, N_plot_min + N_plot):
  # physically normalize WF (norm = 1)
  WF_norm = simps(np.abs(psi[:, i])**2, x=x_vals)
  psi[:, i] /= np.sqrt(WF_norm)
  # higher energy --> higher offset in plotting
  # WF_plot =  WF_scale_factor*np.abs(psi[:,i])**2 + E[i]  # also try plotting real part of WF!
  WF_plot = WF_scale_factor * psi[:, i]  # + E[i]

  plt.plot(x_vals, WF_plot, lw=1.5, label='E = ' + '%.3f' % (E[i]))
  # print("E[%s] = %s"%(i, E[i]))
  print(abs(E[0] + 0.125))
  sys.exit(0)

plt.xlabel("$x$")
plt.legend()
plt.title('Fourier Grid Hamiltonian method')
plt.show()
sys.exit(0)  # ----------------------

# open boundary convergence -----------------------

th_val = -0.125000000000000

N_list = [40, 50, 70, 80, 100, 130, 150, 180, 200]

E_error = []
for N in N_list:
  x_vals = np.linspace(-30., 30., N, endpoint=False)
  V_sampled = [V(x) for x in x_vals]
  H_sampled = H(60., V_sampled)

  E, psi = eigh(H_sampled)

  error = (E[0] - th_val)  #/ (10 ** -8)
  E_error.append(error)
  print(error)

log_N = [x * np.log(x) for x in N_list]
log_E = [np.log(-x) for x in E_error]
print((log_E[0] - log_E[-1]) / (log_N[0] - log_N[-1]))

b, p = polyfit(log_N, log_E, 1)
r2 = '%.4f' % (rsquared(log_N, log_E))
yfit = [b + p * xi for xi in log_N]
p = '%.4f' % (p)

ax.plot(log_N, yfit, alpha=0.4, label='a = ' + p + ', $r^2$ = ' + r2)

ax.plot(log_N, log_E, marker='o', linestyle='None')
#ax.plot(N_list, E_error)

ax.set_xlabel("$N$ log($N$)", fontsize=18)
ax.set_ylabel("log(Error) (au)", fontsize=18)
plt.legend(fontsize=16)
plt.title('Error in ground state vs. grid size \n (FGH method)', fontsize=20)
plt.grid()
plt.show()

sys.exit(0)  # -----------------------

# open boundary convergence (interval) -----------------------

th_val = -0.125000000000000

int_list = [23, 26, 30, 40, 60]

E_error = []
for L in int_list:
  x_vals = np.linspace(-L / 2, +L / 2, L * 20, endpoint=False)
  V_sampled = [V(x) for x in x_vals]
  H_sampled = H(L, V_sampled)

  E, psi = eigh(H_sampled)

  error = (E[0] - th_val) / (10**-6)
  E_error.append(error)
  print(error)
ax.plot(int_list, E_error)

ax.set_xlabel("Interval size, $L$", fontsize=18)
ax.set_ylabel("Error ($10^{-6}$ au)", fontsize=18)
plt.legend(fontsize=16)
plt.title('Error in ground state vs. interval size \n (FGH method)',
          fontsize=20)
plt.grid()
plt.show()

sys.exit(0)  # -----------------------

# periodic boundary -----------------------
N = 2**8
x_vals = np.linspace(0, 2. * pi, 2 * N + 1, endpoint=False)
V_sampled = [V_per(x) for x in x_vals]

H_sampled = H_per(V_sampled)

E, psi = eigh(H_sampled)

N_plot_min = 0  # quantum number of first eigenfunction to plot
N_plot = 3  # number of eigenfunctions to plot

WF_scale_factor = (np.max(V_sampled) - np.min(V_sampled)) / N_plot
plt.plot(x_vals, V_sampled, ls="-", c="k", lw=2, label="$V(x)$")

for i in range(N_plot_min, N_plot_min + N_plot):
  # physically normalize WF (norm = 1)
  WF_norm = simps(np.abs(psi[:, i])**2, x=x_vals)
  psi[:, i] /= np.sqrt(WF_norm)
  # higher energy --> higher offset in plotting
  # WF_plot =  WF_scale_factor*np.abs(psi[:,i])**2 + E[i]  # also try plotting real part of WF!
  WF_plot = WF_scale_factor * psi[:, i]  # + E[i]

  plt.plot(x_vals, WF_plot, lw=1.5, label='E = ' + '%.3f' % (E[i]))
  print("E[%s] = %s" % (i, E[i]))

plt.xlabel("$x$")
plt.legend()
plt.title('Fourier Grid Hamiltonian method')
plt.show()

sys.exit(0)  # -----------------------
