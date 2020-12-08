"""
This module includes functions for the ODE model.
"""

import numpy as np
from scipy.integrate import solve_ivp

def ode_simulation(T, b, k, y0=None, i0=0):
    """
    This function use ODE to model how a new disease spreads throughout a population.
    Inputs:
        i0: the fraction of the infectious population at time 0
        T: number of iterations
        b: the number of interactions each day that could spread the disease (per individual)
        k: the fraction of the infectious population which recovers each day
        y0: if not provided, then initialized as from i0
    Return:
        s: the fraction of the susecptible population at time t
        i: the fraction of the infectious population at time t
        r: the fraction of the removed population at time t
    """

    fun = lambda t, y: [-b * y[0] * y[1], b * y[0] * y[1] - k * y[1], k * y[1]]

    if y0 is None:
        y0 = np.array([1 - i0, i0, 0])
    t_span = (0, T)
    t_eval = range(T+1)

    sol = solve_ivp(fun, t_span, y0, t_eval=t_eval)
    sirs = np.asarray(sol.y).transpose()

    return sirs

if __name__ =='main':
    import pandas as pd

    T = 100
    b = 10
    k = 0.1
    i0 = 0.001
    sirs = ode_simulation(T=T, b=b, k=k, i0=i0)
    pd.DataFrame(sirs, columns=['S','I','R']).plot(legend=True)

    sirs = ode_simulation(T=T, b=b, k=k, y0=[0,0,1])
    pd.DataFrame(sirs, columns=['S','I','R']).plot(legend=True)
