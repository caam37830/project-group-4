import numpy as np
from agent_pop import Population
from typing import Union


class PopulationVarparm(Population):
    """
    This module improves an agent-based model with the following variation of parameters
        p: a susceptible individual becomes infected with probability p
            when he contacts an infectious individual. can change along time
        b: contacts per day can change along time
        k: recover fraction can change along time
    """

    def __init__(self, N, i0):
        super().__init__(N, i0)


    def infect(self, b: int, p: float = 1):
        """
        each infected individual contact b other individuals
        if a susceptible individual is contacted, then he becomes infected with probability p
        """
        contact = np.asarray([np.random.choice(self.N, b, replace=False) for i in self.i_ind]).flatten()
        s_contact_ind = self.s_ind[np.isin(self.s_ind, contact)]
        new_i_ind = s_contact_ind[np.random.binomial(1, p, len(s_contact_ind)) == 1]
        self.s_ind = np.setdiff1d(self.s_ind, new_i_ind, True)
        self.i_ind = np.concatenate((self.i_ind, new_i_ind))


    def simulation(self, T: int, b: Union[int, np.array], k: Union[float, np.array], p: Union[float, np.array]):
        """
        simulate the spread for T days
        return:
            sirs, shape (T, 3), record of daily s,i,r
        """
        sirs = np.array(self.count_sir())
        if type(b) == int:
            b = [b] * T
        if type(k) == float:
            k = [k] * T
        if type(p) == float:
            p = [p] * T
        for t, b_, k_, p_ in zip(range(T),b,k,p):
            self.infect(b_, p_)
            self.recover(k_)
            sirs = np.vstack((sirs, self.count_sir()))

        return sirs


if __name__ == 'main':
    import pandas as pd
    pop = PopulationVarparm(1000, 0.001)

    sirs = pop.simulation(T=100,
                        b=[10]*10 + [5]*10 + [1]*80, # social distancing
                        k=[0.01]*50 + [0.1]*30 + [0.5]*20, # drug developed and distributed
                        p=[0.1]*20 + [0.8]*80) # mutation, become more infectious
    pd.DataFrame(sirs, columns=['S','I','R']).plot(legend=True)
    print(sirs[:, 1])
