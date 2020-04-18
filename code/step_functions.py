import numpy as np
from numba import jit

@jit(nopython=True)
def euler_maruyama(q, h, force, beta, U):
	"""
	Euler-Maruyama sampling scheme
	"""

	q_t = q + h*force(q) + np.sqrt(2*h/beta) * np.random.randn(len(q))

	return q_t


@jit(nopython=True)
def random_walk_metropolis():
	pass


@jit(nopython=True)
def metropolis_adjusted_langevin(q0, h, force, beta, U):
	"""
	MALA sampling scheme
	"""
	
	q_t = euler_maruyama(q0, h, force, beta, U)

    	if np.random.uniform(0,1) < rho(q_t, U)/rho(q0, U):
        	return q_t
    
    	return q0


@jit(nopython=True)
def rho(q, U):
	"""
	Unnormalized probability distribution used for the Metropolis-Hastings step in MALA
	"""
	
        return np.exp(-U(q))
