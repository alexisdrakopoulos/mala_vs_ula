import numpy as np
from numba import jit

@jit(nopython=True)
def euler_maruyama(q, h, force, beta):
	"""
	Euler-Maruyama sampling scheme
	"""

	q_t = q + h*force(q) + np.sqrt(2*h/beta) * np.random.randn(len(q))

	return q_t


@jit(nopython=True)
def random_walk_metropolis():
	pass


@jit(nopython=True)
def metropolis_adjusted_langevin(q0, h, force, beta, step_function):
	"""
	MALA sampling scheme
	"""
	
	q_t = step_function(q0, h, force, beta)

    	if np.random.uniform(0,1) < rho(q_t)/rho(q0):
        	return q_t
    
    	return q0


@jit(nopython=True)
def rho(q):
	"""
	Unnormalized probability distribution
	"""
	
    return np.exp(-U(q))
