import numpy as np
from typing import Union

class Population():
    def __init__(self, N: int, i0: float):
        """
        N: int, population size
        i0: initial infectious fraction i0 in (0,1).
        state: an array of integers 0, 1, 2, corresponds to status s, i, r
        """
        self.N = N
        self.state = np.zeros(N)
        if not 0 < i0 < 1:
            raise ValueError('initial infectious fraction i0 should be in (0,1)')
        i_ind = np.random.choice(N, round(i0*N), replace=False)
        self.state[i_ind] = 1


    def infect(self, b: int):
        """
        each infected individual contact b other individuals
        if a susceptible individual is contacted, then he becomes infected
        """
        I = sum(self.state == 1)
        contact = np.asarray([np.random.choice(self.N, b, replace=False) for i in range(I)]).flatten()
        s_ind = np.argwhere(self.state == 0).flatten()
        new_i_ind = s_ind[np.isin(s_ind,contact)]
        self.state[new_i_ind] = 1

    def recover(self, k: float):
        """
        k fraction of the infectious population recover
        """
        i_ind = np.argwhere(self.state == 1).flatten()
        self.state[i_ind] = np.where(np.random.rand(len(i_ind))<k, 2, self.state[i_ind])


    def count_sir(self):
        s = sum(self.state == 0)/self.N
        i = sum(self.state == 1)/self.N
        r = sum(self.state == 2)/self.N
        return np.array([s,i,r])


    def simulation(self, b: Union[int, np.array], k: Union[float, np.array], T: int):
        """
        simulate the spread for T days
        return:
            sirs, shape (T, 3), record of daily s,i,r
        """

        sirs = np.array(self.count_sir())
        if type(b) == int and type(k) == float: # basic case
            for t in range(T):
                self.infect(b)
                self.recover(k)
                sirs = np.vstack((sirs, self.count_sir()))
        else:
            if type(b) == int:
                b = [b] * T
            if type(k) == float:
                k = [k] * T
            for b_, k_, t in zip(b,k,range(T)):
                self.infect(b_)
                self.recover(k_)
                sirs = np.vstack((sirs, self.count_sir()))

        return sirs


if __name__ == 'main':
    pop = Population(1000, 0.1)
    sirs = pop.simulation(b=[9]*100, k=0.01, T=100)
    import pandas as pd
    pd.DataFrame(sirs, columns=['S','I','R']).plot(legend=True)
    print(sirs)
