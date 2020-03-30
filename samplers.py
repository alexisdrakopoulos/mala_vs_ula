import numpy as np
from numba import jit
from tqdm import tqdm
from code.step_functions import euler_maruyama
from code.step_functions import metropolis_hastings
from code.step_functions import metropolis_adjusted_langevin


@jit(nopython=True)
def GaussianForce(q):
    
    # Force = -d_dq U(q)
    # If U(q) = q^2/2, then...
    f = -q 
    return f


@jit(nopython=True)
def run_simulation(q0, N, h, beta, step_function, force):
    
    # initialize q trajectory and build t traj
    q_traj = np.zeros(N+1).reshape(N+1, 1)
    q_traj[0] = q0
    t_traj = np.arange(0, h*N, h)

    # now compute the steps
    for i in range(1, N+1):
        q_traj[i] = step_function(q_traj[i-1], h, force, beta)

    return q_traj, t_traj


if __name__ == "__main__":
    q0 = np.zeros(1)
    N = 10_000
    q, t = run_simulation(q0, N, 0.1, 1, euler_maruyama, GaussianForce)
    print(q)