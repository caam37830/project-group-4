import sys
sys.path.append('../sir')
from agent_varparm import PopulationVarparm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy

def plot_sir(sirs, sirs2=None, event_time=None, save_as=None):
    """
    event_time: when sirs2 start to differ from sirs
    """
    T = len(sirs)
    fig, ax = plt.subplots()
    ax.plot(range(T), sirs[:,0], label = r'$s$')
    ax.plot(range(T), sirs[:,1], label = r'$i$')
    ax.plot(range(T), sirs[:,2], label = r'$r$')
    ax.set_xlabel('Time')
    ax.set_ylabel('Fraction')

    if sirs2 is not None:
        ax.plot(range(event_time,T), sirs2[:,0], label = r'$s^\prime$', c='C0', linestyle='--')
        ax.plot(range(event_time,T), sirs2[:,1], label = r'$i^\prime$', c='C1', linestyle='--')
        ax.plot(range(event_time,T), sirs2[:,2], label = r'$r^\prime$', c='C2', linestyle='--')

    fig.legend(bbox_to_anchor=(1,0.5), loc="center left", borderaxespad=0)

    if save_as:
        fig.savefig(f'../output/{save_as}.png', dpi=300, bbox_inches='tight')


# hyper parm
N = 10000
i0 = 0.001
T = 100

# base case
b, k, p = 10, 0.01, 0.1
pop = PopulationVarparm(N,i0)
sirs = pop.simulation(T=T, b=b, k=k, p=p)
plot_sir(sirs, save_as='agent_varparm_base_case')


# social distancing
pop0 = PopulationVarparm(N,i0)
bprds = [5, 25, 70]
bs = [10, 3, 1]
sirs = pop0.simulation(T=bprds[0], b=bs[0], k=k, p=p)
pop1 = copy.copy(pop0)
sirs_dist = copy.copy(sirs[-1])
for prd, b_ in zip(bprds[1:], bs[1:]):
    sirs = np.vstack((sirs, pop0.simulation(T=prd, b=bs[0], k=k, p=p)[1:]))
    sirs_dist = np.vstack((sirs_dist, pop1.simulation(T=prd, b=b_, k=k, p=p)[1:]))

plot_sir(sirs, sirs_dist, bprds[0], 'agent_varparm_social_distancing')

# drug developement and distribution
pop0 = PopulationVarparm(N,i0)
kprds = [15, 55, 30]
ks = [0.01, 0.1, 0.8]
sirs = pop0.simulation(T=kprds[0], b=b, k=ks[0], p=p)
pop1 = copy.copy(pop0)
sirs_drug = copy.copy(sirs[-1])
for prd, k_ in zip(kprds[1:], ks[1:]):
    sirs = np.vstack((sirs, pop0.simulation(T=prd, b=b, k=ks[0], p=p)[1:]))
    sirs_drug = np.vstack((sirs_drug, pop1.simulation(T=prd, b=b, k=k_, p=p)[1:]))

plot_sir(sirs, sirs_drug, kprds[0], 'agent_varparm_drug')

# virus mutation
pop0 = PopulationVarparm(N,i0)
pprds = [20, 80]
ps = [0.1, 0.8]
sirs = pop0.simulation(T=pprds[0], b=b, k=k, p=ps[0])
pop1 = copy.copy(pop0)
sirs_mut = copy.copy(sirs[-1])
for prd, p_ in zip(pprds[1:], ps[1:]):
    sirs = np.vstack((sirs, pop0.simulation(T=prd, b=b, k=k, p=ps[0])[1:]))
    sirs_mut = np.vstack((sirs_mut, pop1.simulation(T=prd, b=b, k=k, p=p_)[1:]))
plot_sir(sirs, sirs_mut, pprds[0], 'agent_varparm_mutation')




# all together
pop = PopulationVarparm(N,i0)
sirs_all = pop.simulation(T=T, b=np.repeat(bs, bprds),
                                k=np.repeat(ks, kprds),
                                p=np.repeat(ps, pprds))

plot_sir(sirs, sirs_all, 0, 'agent_varparm_all')
