import sys
sys.path.append('../sir')
import numpy as np
from scipy.integrate import solve_ivp

def diff_matrix(M):
    """
    returns the second-order difference matrix
    """
    res = np.zeros((M,M))
    res[0,:2] = [-2,2]
    for i in range(1, M-1):
        res[i,i-1:i+2] = [1,-2,1]
    res[M-1,-2:] = [2,-2]
    return res

def pde_model(i0, pos, M, T, b, k, p):
    """
    Inputs:
        i0: the fraction of the infectious population at time 0
        pos: the position (i,j) of the initial infectious population
        M: the number of grids on [0,1]
        T: the number of iterations
        b: the number of interactions each day that could spread the disease (per individual)
        k: the fraction of the infectious population which recovers each day
        p: parameter for diffusion
    """

    D = diff_matrix(M)

    def f(t, y):
        y = y.reshape((3,M,M))
        res = np.zeros((3,M,M))
        res[0] = -b * y[0] * y[1] + (D @ y[0] + y[0] @ D.T) * p * M**2 # p / h**2
        res[1] = b * y[0] * y[1] - k * y[1] + (D @ y[1] + y[1] @ D.T) * p * M**2
        res[2] = k * y[1] + (D @ y[2] + y[2] @ D.T) * p * M**2
        return res.flatten()

    y0 = np.zeros((3,M,M))
    y0[0] = 1                 # all people are susceptible
    y0[1,pos[0],pos[1]] = i0  # initial infection
    y0 = y0.flatten()

    t_span = (0, T)
    t_eval = np.linspace(0, T, int(T)+1)

    sol = solve_ivp(f, t_span, y0, t_eval=t_eval, vectorized=True, atol=1e-8)
    return sol.y.reshape((3,M,M,int(T)+1))
