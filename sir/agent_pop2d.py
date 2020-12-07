import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from time import time
import pandas as pd

class Population2d():
    """
    This module implements an 2d agent-based model.
    We consider the spatial position and movement of the agents
    An infectious individual can infect his neighbors with certain distance
    """
    def __init__(self, N: int, I: int):
        self.N = N
        self.i_info = np.random.rand(I, 3)
        self.s_info = np.random.rand(N-I, 3)
        self.i_info[:,0] = np.random.choice(N, I, replace=False)
        self.s_info[:,0] = np.setdiff1d(np.arange(N), self.i_info[:,0], True)
        self.r_info = np.array([])


    def move(self, p):
        """
        each agent move a step of length p in a random direction
        if the new position is outside the unit squaure, use the old position
        """
        for info in [self.i_info, self.s_info, self.r_info]:
            if info.size == 0:
                continue
            pos = info[:, 1:]
            dpos = np.random.randn(*pos.shape)
            dpos = dpos / np.linalg.norm(dpos, axis=1, keepdims=True)
            new_pos = pos + dpos*p
            inside = np.all([0<=new_pos[:,0], new_pos[:,0]<=1,
                             0<=new_pos[:,1], new_pos[:,1]<=1],
                             axis=0)
            info[:,1:] += np.where(inside[:,np.newaxis], dpos*p, 0) # in-place


    def infect(self, q):
        """
        an infectious individual infect his susceptible neighbors within distance q
        """
        X = np.vstack((self.s_info,self.i_info))
        if X.size == 0:
            # print('all individuals are recovered')
            return
        tree = KDTree(X[:,1:]) # pos of S, I
        contact_tree_inds = tree.query_ball_point(self.i_info[:,1:], q) # a list of lists of indices
        contact_tree_inds = sum(contact_tree_inds, []) # flatten a list of lists
        contact_inds = X[:,0][contact_tree_inds]
        s_contact_slice = np.isin(self.s_info[:,0], contact_inds)
        new_i_info = self.s_info[s_contact_slice]
        self.s_info = self.s_info[~s_contact_slice]
        self.i_info = np.concatenate((self.i_info, new_i_info))

    def recover(self, k):
        if self.i_info.size == 0:
            # print('no infectious individuals to recover')
            return
        new_r_slice = np.random.binomial(1, k, len(self.i_info)) == 1
        new_r_info = self.i_info[new_r_slice]
        self.i_info = self.i_info[~new_r_slice]
        if self.r_info.size == 0:
            self.r_info = new_r_info
        else:
            self.r_info = np.concatenate((self.r_info, new_r_info))


    def count_SIR(self):
        return np.array([len(self.s_info), len(self.i_info), len(self.r_info)])


    def scatter_plot(self, title=None):
        plt.scatter(self.s_info[:,1], self.s_info[:,2])
        plt.scatter(self.i_info[:,1], self.i_info[:,2])
        if self.r_info.size > 0:
            plt.scatter(self.r_info[:,1], self.r_info[:,2])
        if title:
            plt.title(title)
        plt.show()


    def simulation(self, T, p, q, k, plot_time_interval=None):
        """
        simulate the spread for T days
        return:
            SIRs, shape (T, 3), record of daily S, I, R
        """
        SIRs = np.array(self.count_SIR())
        ts = []
        # self.scatter_plot('begin')
        for t in range(T):
            t0 = time()

            self.move(p)
            t1 = time()

            self.infect(q)
            t2 = time()

            self.recover(k)
            t3 = time()

            SIRs = np.vstack((SIRs, self.count_SIR()))
            ts.append([t1 - t0, t2 - t1, t3 - t2])

            if plot_time_interval and t % plot_time_interval == 0:
                self.scatter_plot(title=f't = {t}, i = {len(self.i_info)/self.N}')

        return SIRs, np.array(ts)





if __name__ == 'main':

    pop = Population2d(1000, 1)
    t0 = time()
    SIRs, ts = pop.simulation(T=10, p=0.2, q=0.1, k=0.1, plot_time_interval=1)
    t1 = time()
    t1 - t0


    # plot SIR by time
    pd.DataFrame(SIRs, columns=['S', 'I', 'R']).plot(legend=True)

    # check which step is time consuming. ans: the infect() method.
    pd.DataFrame(ts, columns=['t(S)','t(I)','t(R)']).plot(legend=True, title='computation time')
