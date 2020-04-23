import numpy as np
import sys
from numba import jit
from tqdm import tqdm
from code.step_functions import euler_maruyama
from code.step_functions import random_walk_metropolis
from code.step_functions import metropolis_adjusted_langevin
from code.example_distributions import *
from scipy.stats import wasserstein_distance
from scipy.stats import multivariate_normal
from multiprocessing import Pool



def compute_distance(q, sample):
    distances = []
    for i in range(q.shape[1]):
        distances.append(wasserstein_distance(q[:,i], sample[:,i]))
    return distances

def n_distance_calc(q, sample, n=10_000):
    distances = []
    for i in range(n, len(q), n):
        distances.append(compute_distance(q[:i], samples))
        
    return distances


def generate_samples(n=100, a=1/20, b=100, mu=1, N=100_000):

    n = 100
    xlims = [-10, 10]
    ylims = [-1, 40]

    samples = np.array([np.array(rhybrid(n, a, [[b]], mu)) for _ in range(N//n)]).reshape(N, 2)

    # limit samples
    samples_1 = samples[:,0][(samples[:,0] > xlims[0]) & (samples[:,0] < xlims[1]) & (samples[:,1] > ylims[0]) & (samples[:,1] < ylims[1])]
    samples_2 = samples[:,1][(samples[:,0] > xlims[0]) & (samples[:,0] < xlims[1]) & (samples[:,1] > ylims[0]) & (samples[:,1] < ylims[1])]
    #samples2 = np.array([samples_1, samples_2]).reshape(2, samples_1.shape[0])
    samples = np.array([samples_1, samples_2]).reshape(2, samples_1.shape[0])

    return samples

def print_statusline(msg: str):
    last_msg_length = len(print_statusline.last_msg) if hasattr(print_statusline, 'last_msg') else 0
    print(' ' * last_msg_length, end='\r')
    print(msg, end='\r')
    sys.stdout.flush()  # Some say they needed this, I didn't.
    print_statusline.last_msg = msg


def run_simulation(q0, N, h, beta, step_function, force, epoch_iter, samples, multiplier=0.9):
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


    # now compute the steps
    overall_d = [np.inf]*len(q0)
    counter = np.zeros(len(q0))
    samp = 10
    initial_h = np.array(h).copy()
    overall_accepted = 0
    accepted_per_epoch = []
    distance_per_epoch = []
    h_values = np.zeros((N+1)*len(q0)).reshape(N+1, len(q0))
    
    # burn in
    print("Performing Burn-in Period to optimize step size")
    i = 0
    reduction_count = 0
    while i < 20_000:
        
        # do the computation
        i += 1
        q_traj[i] = step_function(q=q_traj[i-1], h=h, force=force, beta=beta, U=U)
        
        
        if i % 500 == 0:
            
            print_statusline(f"Step: {i}, Rate: {np.round(h,4)}, Distance: {overall_d}")
            
            # make sure that the number is valid and no over/under flow happened
            # if under/overflow happened decrease step size and try again
            if np.any(np.isnan(q_traj[i-500:])) or np.any(np.isinf(q_traj[i-500:])):
                q_traj[:i-500] = 0
                i -= 500
                reduction_count += 1
                h *= multiplier

        if i % 2000 == 0 and i > 1999:
                overall_d = compute_distance(q_traj[i-2000:i], sample=samples)
                h[np.array(overall_d) > 100] *= 0.9

                # Count % accepted/rejected
                accepted = np.unique(q_traj[i-2000:i], axis = 0).shape[0]

                # if acceptance rate < 5% lower learning rates
                if accepted/2000 < 0.05:
                    i -= 2000
                    reduction_count += 1
                    h *= multiplier
        
        # if algo stuck restart
        if reduction_count > 300:
            print("Issue converging, restarting model")
            reduction_count = 0
            i = 0
            h = initial_h.copy() * np.random.uniform(0.2, 0.9)
            q_traj = np.zeros((N+1)*len(q0)).reshape(N+1, len(q0))
            q_traj[i] = np.zeros(q_traj.shape[1]) + np.random.randn(q_traj.shape[1])
    

    print("\n")
    print("Starting MCMC Sampling")
    # now the real computations start
    for i in range(1, N+1):
        if i % epoch_iter == 0:
            # amount of samples whose distance to compute:
            if i > 1_000_000:
                samp = 100
            elif i > 5_000_000:
                samp = 250
            elif i > 10_000_000:
                samp = 500
                
            # compute distance for this epoch
            curr_d = compute_distance(q_traj[:i-1][::i//samp], samples)
            distance_per_epoch.append(curr_d)
            
            # Count % accepted/rejected
            accepted = np.unique(q_traj[i-epoch_iter:i], axis = 0).shape[0]
            accepted_per_epoch.append(accepted)
            overall_accepted += accepted
            
            # if acceptance rate < 5% lower learning rates
            if overall_accepted/i < 0.05:
                h *= multiplier
            
            # print and wipe line
            print_statusline(f"Step: {i}, Counter: {counter}, % Accepted: {round(overall_accepted/i, 2)} Rate: {np.round(h,4)}, Distance: {np.round(np.array(curr_d),2)}, Best Distance: {np.round(np.array(overall_d),2)}")
            
            # if our distance gets worse increment counter by 1
            counter[np.array(curr_d) < np.array(overall_d)] += 1
            overall_d = curr_d
            
            # if any of the counters have gotten worse for 3 times decrease their step size
            if np.any(counter == 3):
                h[counter == 3] *= multiplier
                
                # early stop if ALL step sizes drop below threshold
                if np.all(h[counter == 3] < 0.00001):
                    print(f"Stopping at {i} iterations...")
                    return q_traj[:i], accepted_per_epoch, distance_per_epoch, h_values
            
                # reset counter after learning rate was decreased and early stop check
                counter[counter == 3] = 0
        
        # do the computation
        q_traj[i] = step_function(q=q_traj[i-1], h=h, force=force, beta=beta, U=U)
        h_values[i] = h

        # make sure that the number is valid and no over/under flow happened
        # if under/overflow happened decrease step size and try again
        if np.any(np.isnan(q_traj[i])) or np.any(np.isinf(q_traj[i])):
            q_traj[i] = np.zeros(q_traj.shape[1])
            h *= multiplier
            

    return q_traj, accepted_per_epoch, distance_per_epoch, h_values


def multithread_run(value):
    q0 = np.zeros(2)
    a = 1/20
    b = 100
    xlims = [-10, 10]
    ylims = [-1, 40]
    mu = 1
    n = 100

    # get samples from distribution
    samples = generate_samples()

    q, accepted_per_epoch, distance_per_epoch, h = run_simulation(q0,
                                                           5_000_000,
                                                           np.array([0.001, 0.001]),
                                                           1,
                                                           metropolis_adjusted_langevin,
                                                           force,
                                                           25_000,
                                                           samples=samples,
                                                           multiplier=1)
    
    return q, accepted_per_epoch, distance_per_epoch, h


if __name__ == "__main__":

    # run simulation
    q0 = np.zeros(2)

    # get the true distribution samples
    a = 1/20
    b = 100
    xlims = [-10, 10]
    ylims = [-1, 40]
    mu = 1
    n = 100

    samples = generate_samples()

    print(f"Starting {metropolis_adjusted_langevin.__name__} experiments:\n")
    with Pool() as p:
        q_vals = p.map(multithread_run, [5]*16)

    print("Saving experiments")
    np.save(f"{metropolis_adjusted_langevin.__name__}_experiments_fixed.npy", np.array(q_vals))

    # # plot
    # X,Y = np.meshgrid( np.linspace(-3,5,200) , np.linspace(-1,11,100) )
    # rho = np.exp(-U(np.array([X, Y])))

    # plt.figure()
    # plt.pcolor(X,Y,rho,vmin=0,vmax=1)
    # plt.scatter(q[:,0], q[:,1], s=1, color="red")
    # plt.xlim(-3, 5)
    # plt.ylim(-1, 11)
    # plt.show()
    # print(q)
    # print(q.shape)