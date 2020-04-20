import numpy as np
from numba import jit


@jit(nopython=True)
def ban_Dist(q):   
      """
      The banana distribution
      
      Args:
          q: 2-dimensional coordinate vector
      """
      
      x = q[0]
      y = q[1]
      return (1-x)**2 + 4*(2*y-x**2)**2
  


@jit(nopython=True)
def ban_Force(q):
      """
      The banana distibution associated force from Hamilton's equations
      """
      
      x = q[0]
      y = q[1]
      dx = 2*(8*x**3-16*x*y+x-1)
      dy = 16 * (2*y-x**2)
      return np.array([-dx,-dy])


@jit(nopython=True)
def Gaussian_Force(q):
      """
      The Guassian Force
      """
      
      # Force = -d_dq U(q)
      # If U(q) = q^2/2, then...
      f = -q 
      return f


@jit(nopython=True)
def Gaussian_Dist(q):
      """
      Guassian Distribution
      """
      
      return q**2/2
