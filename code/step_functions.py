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
def metropolis_adjusted_langevin(q0, h, force, beta, U, step_function=euler_maruyama):
	"""
	MALA sampling scheme
	"""
	
	q_t = step_function(q0, h, force, beta)

    	if np.random.uniform(0,1) < rho(q_t, U)/rho(q0, U):
        	return q_t
    
    	return q0


@jit(nopython=True)
def rho(q, U):
	"""
	Unnormalized probability distribution
	"""
	
        return np.exp(-U(q))


@jit(nopython=True)
def run_simulation(q0, N, h, beta, step_function, force, U=None, stepScheme=None):
   	"""
	Run sampling simulation
	
	Args:
	    U: The target distribution required for the metropolis hastings step in MALA.
	"""
	
    	q_traj = np.zeros((len(q0),N+1))
    	q_traj[:,0] = q0
    	t_traj = np.arange(0, h*N, h)
     
    	if stepScheme is None:
        	for i in range(1, N+1):
            		q_traj[:,i] = step_function(q_traj[:,i-1], h, force, beta, U)
    	else:
        	for i in range(1, N+1):
            		q_traj[:,i] = step_function(q_traj[:,i-1], stepScheme(h,i,N), force, beta, U)
 
    	return q_traj, t_traj
