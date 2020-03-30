import numpy as np
from numba import jit

@jit(nopython=True)
def euler_maruyama_step(q, h, force, beta):
	"""
	Euler-Maruyama sampling scheme
	"""

	q_t = q + h*force(q) + np.sqrt(2*h/beta) * np.random.randn(len(q))

	return q_t


@jit(nopython=True)
def random_walk_metropolis():
	pass


@jit(nopython=True)
def metropolis_adjusted_metropolis():
	pass