import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numba import jit


@jit(nopython=True)
def U(q, a=1/20, b=100):
    x = q[0]
    y = q[1]
    return ((1 - x)**2 + b*(y - x**2)**2)*a


@jit(nopython=True)
def force(q, a=1/20, b=100):
    x = q[0]
    y = q[1]
    Fx = -2*a*(1-x) - 4*b*a*x*(y-x**2)
    Fy = 2*b*a*(y - x**2)
    
    return -np.array([Fx, Fy])



def rhybrid(n, a, b, mu):
    
    blocks = len(b)
    blocklengths = [len(i) for i in b]
    
    mat = np.zeros([n, 1])
    mat[:,0] = np.random.normal(mu, 1/(2*a), n)
    
    for j in range(blocks):
        for i in range(blocklengths[j]):
            if i == 1:
                prev = mat[:,0]
            else:
                prev = mat[:,-1]

            cal = np.random.multivariate_normal(prev**2,
                                                np.diag([1]*n)/(2*b[j][i]),
                                                size=1)
            mat = np.concatenate((mat, cal.reshape([n, 1])), axis = 1)

    return mat


def hybrid_pdf(xprime, x, a, b, mu):
    
    n2 = len(x)
    n1 = len(x[0])
    
    const = np.pi**((n2*(n1-1)+1)/2) / (np.sqrt(a)*np.prod([np.prod(np.sqrt(x)) for x in b]))

    def inner_prod(z, p):
        z = np.array(z)
        return np.sum(p*(z[1:] - z[:-1]**2)**2)
    
    return np.exp(-(a*(xprime - mu)**2 + np.sum([i for i in map(inner_prod, x, b)])))