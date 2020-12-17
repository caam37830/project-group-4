"""
This module includes functions for the ODE model.
"""
import sys
sys.path.append('../sir')
import numpy as np
from scipy.integrate import solve_ivp

def ode_simulation(e0, i0, T, b, f, k):
    """
    This function use ODE to model how a new disease spreads throughout a population.
    Inputs:
        i0: the fraction of the infectious population at time 0
        T: number of iterations
        b: the number of interactions each day that could spread the disease (per individual)
        f: the fraction of the exposed individuals that becomes infectious each day
        k: the fraction of the infectious population which recovers each day
    Return:
        seirs, shape (T+1, 4), evaluation of s,e,i,r at range(T+1)
    """

    fun = lambda t, y: [-b * y[0] * y[1], b * y[0] * y[1] - f * y[1], f * y[1] - k * y[2], k * y[2]]

    y0 = np.array([1 - i0 - e0, e0, i0, 0])
    t_span = (0, T)
    t_eval = range(T+1)

    sol = solve_ivp(fun, t_span, y0, t_eval=t_eval)
    seirs = np.asarray(sol.y).transpose()

    return seirs


if __name__ == 'main':
    e0 = i0 = 0.001
    T = 100
    b = 7
    f = 0.2
    k = 0.2

    import pandas as pd
    seirs = ode_simulation(e0, i0, T, b, f, k)
    pd.DataFrame(seirs, columns=['S','E','I','R']).plot(legend=True)
