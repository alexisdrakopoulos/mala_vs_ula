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
def metropolis_adjusted_langevin(q0, h, force, beta, step_function=euler_maruyama_step):
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


@jit(nopython=True)
def run_simulation(q0, N, h, beta, step_function, force, stepScheme=None):
   	"""
	Run sampling simulation
	"""
	
    	q_traj = np.zeros((len(q0),N+1))
    	q_traj[:,0] = q0
    	t_traj = np.arange(0, h*N, h)
     
    	if stepScheme is None:
        	for i in range(1, N+1):
            		q_traj[:,i] = step_function(q_traj[:,i-1], h, force, beta)
    	else:
        	for i in range(1, N+1):
            		q_traj[:,i] = step_function(q_traj[:,i-1], stepScheme(h,i,N), force, beta)
 
    	return q_traj, t_traj
