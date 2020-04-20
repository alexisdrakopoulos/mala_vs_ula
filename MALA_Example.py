import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from code.step_functions import metropolis_adjusted_langevin
from code.distributions import ban_Force, ban_Dist


@jit(nopython=True)
def run_simulation(q0, N, h, beta, step_function, force, U=None, stepScheme=None):
    """
    Run sampling simulation

	Args:
        q0: An n-dimensional list of initial starting values.
        M: Maximum number of steps.
        h: Step size.
        beta: Scaling factor?
        U: The target distribution required for the Metropolis-Hastings step in MALA.
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


if __name__ == "__main__":
    N = 10_000
    q0 = np.array([1.0,1.0/2])
    U = ban_Dist
    plt.figure(figsize=(10,5))
    q1, t1 = run_simulation(q0, N, 0.02, 0.1, metropolis_adjusted_langevin, ban_Force, U)
    plt.plot(q1[0,:],q1[1,:],'g.',label='ULA')
    plt.legend()
    X,Y = np.meshgrid( np.linspace(-3,5,200) , np.linspace(-1,11,100) )
    plt.pcolor(X,Y,U([X,Y]),vmin=0,vmax=15,cmap='Greys_r')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title('The Rosenbrock function')
    plt.show()
