import numpy as np
from typing import Union

class Population():
    """
    This module implements an agent-based model.
    It defines a class which represents a population of size N, with an internal state which is
    one of S, I or R.

    By default, a person is susceptible.  They become infected using the infect method,
    and recovered by recover method.
    """

    def __init__(self, N: int, i0: float):
        """
        input:
            N: int, population size
            i0: initial infectious fraction i0 in (0,1).
        values:
            store S, I, R individuals indexes in i_ind, s_ind, and r_ind.
        """
        self.N = N
        if not 0 < i0 < 1:
            raise ValueError('initial infectious fraction i0 should be in (0,1)')
        self.i_ind = np.random.choice(N, round(i0*N), replace=False)
        self.s_ind = np.setdiff1d(np.arange(N), self.i_ind, True)
        self.r_ind = np.array([])


    def infect(self, b: int):
        """
        each infected individual contact b other individuals
        if a susceptible individual is contacted, then he becomes infected
        for simplicity, we don't want to loop over each infected individuals, when I is large
        consider from the population perspective,
        the probability for an individual not to be contact by an I is (N-1-b)/(N-1)
        so the probability for an individual not to be contact by all I's is [(N-b)/(N-1)]^len(I)
        """
        # contact = np.asarray([np.random.choice(self.N, b, replace=False) for i in self.i_ind]).flatten()
        # new_i_ind = self.s_ind[np.isin(self.s_ind, contact)]
        p_contact = 1 - ((self.N-1-b)/(self.N-1))**len(self.i_ind)
        new_i_ind = np.random.choice(self.s_ind, round(p_contact*len(self.s_ind)), replace=False)
        self.s_ind = np.setdiff1d(self.s_ind, new_i_ind, True)
        self.i_ind = np.concatenate((self.i_ind, new_i_ind))


    def recover(self, k: float):
        """
        k fraction of the infectious population recover
        """
        new_r_ind = self.i_ind[np.random.binomial(1, k, len(self.i_ind)) == 1]
        self.i_ind = np.setdiff1d(self.i_ind, new_r_ind, True)
        self.r_ind = np.concatenate((self.r_ind, new_r_ind))


    def count_sir(self):
        return np.array([len(self.s_ind), len(self.i_ind), len(self.r_ind)])/self.N


    def simulation(self, T: int, b: int, k: float):
        """
        simulate the spread for T days
        return:
            sirs, shape (T, 3), record of daily s,i,r
        """
        sirs = np.array(self.count_sir())
        for t in range(T):
            self.infect(b)
            self.recover(k)
            sirs = np.vstack((sirs, self.count_sir()))

        return sirs


if __name__ == 'main':
    import pandas as pd
    from time import time

    pop = Population(1000, 0.001)
    t0 = time()
    sirs = pop.simulation(T=100, b=5, k=0.5)
    t1 = time()
    t1 - t0

    pd.DataFrame(sirs, columns=['S','I','R']).plot(legend=True)
