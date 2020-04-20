import numpy as np
from numba import jit
from code.example_distributions import U

#@jit(nopython=True)
def euler_maruyama(q, h, force, beta=1):
	"""
	Euler-Maruyama sampling scheme
	"""

	q_t = q + h*force(q) + np.sqrt(2*h/beta) * np.random.randn(len(q))

	return q_t



#@jit(nopython=True)
def random_walk_metropolis(q, h, force, U=U):
    
    q_t = q + h*np.random.normal(size=len(q))
    r = np.random.uniform(0,1)
    diff = np.exp(-U(q_t))/np.exp(-U(q))
    if r < diff:
        return q_t
    return q


#@jit(nopython=True)
def metropolis_adjusted_langevin(q, h, force, U=U, beta=1):
    
    q_t = euler_maruyama(q, h, force, beta)
    r = np.random.uniform(0,1)
    diff = np.exp(-U(q_t))/np.exp(-U(q))
    if r < diff:
        return q_t
    return q