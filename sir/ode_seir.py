"""
This module includes functions for the ODE model.
"""

import numpy as np
from scipy.integrate import solve_ivp

def ode_model(i0, T, a, b, k):
    """
    This function use ODE to model how a new disease spreads throughout a population.
    Inputs:
        i0: the fraction of the infectious population at time 0
        T: number of iterations
        a: the fraction of the exposed individuals that befomes infectious each day
        b: the number of interactions each day that could spread the disease (per individual)
        k: the fraction of the infectious population which recovers each day
    Return:
        s: the fraction of the susecptible population at time t
        e: the fraction of the exposed population at time t
        i: the fraction of the infectious population at time t
        r: the fraction of the removed population at time t
    """

    f = lambda t, y: [-b * y[0] * y[1], b * y[0] * y[1] - a * y[2], a * y[2] - k * y[1], k * y[1]]

    y0 = np.array([1 - i0, 0, i0, 0])
    t_span = (0, T)
    t_eval = np.linspace(0, T, int(T))

    sol = solve_ivp(f, t_span, y0, t_eval=t_eval)

    s, e, i, r = sol.y
    return s, e, i, r
