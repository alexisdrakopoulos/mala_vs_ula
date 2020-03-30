import numpy as np
from numba import jit
from tqdm import tqdm
from code.step_functions import euler_maruyama
from code.step_functions import random_walk_metropolis
from code.step_functions import metropolis_adjusted_langevin


@jit(nopython=True)
def GaussianForce(q):
    
    # Force = -d_dq U(q)
    # If U(q) = q^2/2, then...
    f = -q 
    return f


@jit(nopython=True)
def run_simulation(q0, N, h, beta, step_function, force):
    """
    general function to run samplers from code.step_functions.py

    Inputs:
        q0 - np.array object with initial values
        N - int of number of steps
        h - float of step size
        beta - float
        step_function - python jit func to use as step function
        force - python jit func to use as force function
    Outputs:
        q_traj - np.array final trajectory
        t_traj - np.array the time trajectory built automatically
    """
    
    # initialize q trajectory and build t traj
    # had to use the (N+1)*len(q0) hack instead of direct shape input
    # due to issues with numba jit
    q_traj = np.zeros((N+1)*len(q0)).reshape(N+1, len(q0))
    q_traj[0] = q0
    t_traj = np.arange(0, h*N, h)

    # now compute the steps
    for i in range(1, N+1):
        q_traj[i] = step_function(q_traj[i-1], h, force, beta)

    return q_traj, t_traj


if __name__ == "__main__":
    q0 = np.zeros(1)
    N = 100_000
    q, t = run_simulation(q0, N, 0.1, 1, euler_maruyama, GaussianForce)
    print(q)
    print(q.shape)