import numpy as np
from agent_pop import Population
from typing import Union


class PopulationSEIR(Population):
    """
    This module improves an agent-based model by adding an additional status E
    An susceptible individual becomes exposed if he is infected,
    but not yet infectious.
    """

    def __init__(self, N, e0, i0):
        super().__init__(N, i0)
        ei_ind = np.random.choice(N, round((e0+i0)*N), replace=False)
        cutoff = round(len(ei_ind)*e0/(e0+i0))
        self.s_ind = np.setdiff1d(np.arange(N), ei_ind, True)
        self.e_ind = ei_ind[:cutoff]
        self.i_ind = ei_ind[cutoff:]
        self.r_ind = np.array([])

    def expose(self, b: int):
        """
        each infected individual contact b other individuals
        if a susceptible individual is contacted, then he becomes exposed
        """
        contact = np.asarray([np.random.choice(self.N, b, replace=False) for i in self.i_ind]).flatten()
        new_e_ind = self.s_ind[np.isin(self.s_ind, contact)]
        self.s_ind = np.setdiff1d(self.s_ind, new_e_ind, True)
        self.e_ind = np.concatenate((self.e_ind, new_e_ind))

    def infect(self, f: float):
        """
        f fraction of the exposed population become infectious each day
        """
        new_i_ind = self.e_ind[np.random.binomial(1, f, len(self.e_ind)) == 1]
        self.e_ind = np.setdiff1d(self.e_ind, new_i_ind, True)
        self.i_ind = np.concatenate((self.i_ind, new_i_ind))

    def count_seir(self):
        return np.array([len(self.s_ind), len(self.e_ind), len(self.i_ind), len(self.r_ind)])/self.N

    def simulation(self, T: int, b: int, f: float, k: float):
        """
        simulate the spread for T days
        return:
            sirs, shape (T, 4), record of daily s,e,i,r
        """
        seirs = np.array(self.count_seir())
        for t in range(T):
            self.expose(b)
            self.infect(f)
            self.recover(k)
            seirs = np.vstack((seirs, self.count_seir()))

        return seirs



if __name__ == 'main':
    import pandas as pd
    from time import time

    pop = PopulationSEIR(1000, 0.002, 0.005)
    t0 = time()
    seirs = pop.simulation(T=100, b=9, f=0.3, k=0.1)
    t1 = time()
    t1 - t0

    pd.DataFrame(seirs, columns=['S','E','I','R']).plot(legend=True)
