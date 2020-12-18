import sys
sys.path.append('../sir')
from ode_varparm import ode_simulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy


# hyper parm
N = 1000
i0 = 0.001
T = 100

def plot_sir(sirs, sirs2=None, event_time=None, title=None, save_as=None):
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
    if title:
        plt.title(title)

    if save_as:
        fig.savefig(f'../docs/figs/{save_as}.png', dpi=300, bbox_inches='tight')


# base case
b, k, p = 10, 0.01, 0.1
power = 1.5

sirs = ode_simulation(T=T, b=b*p**power, k=k, i0=i0)
plot_sir(sirs, title='Base case', save_as='ode_varparm_base_case')

# social distancing
bprds = [5, 25, 70]
bs = [10, 3, 1]
sirs = ode_simulation(T=bprds[0], b=bs[0]*p**power, k=k, i0=i0)
sirs_dist = copy.copy(sirs[-1])
sirs_dist = sirs_dist[np.newaxis, :]
for prd, b_ in zip(bprds[1:], bs[1:]):
    sirs = np.vstack((sirs, ode_simulation(T=prd, b=bs[0]*p**power, k=k, y0=sirs[-1])[1:]))
    sirs_dist = np.vstack((sirs_dist, ode_simulation(T=prd, b=b_*p**power, k=k, y0=sirs_dist[-1])[1:]))

plot_sir(sirs, sirs_dist, bprds[0], title='Early social distancing', save_as='ode_varparm_social_distancing_early')

bprds = [20, 50, 70]
bs = [10, 3, 1]
sirs = ode_simulation(T=bprds[0], b=bs[0]*p**power, k=k, i0=i0)
sirs_dist = copy.copy(sirs[-1])
sirs_dist = sirs_dist[np.newaxis, :]
for prd, b_ in zip(bprds[1:], bs[1:]):
    sirs = np.vstack((sirs, ode_simulation(T=prd, b=bs[0]*p**power, k=k, y0=sirs[-1])[1:]))
    sirs_dist = np.vstack((sirs_dist, ode_simulation(T=prd, b=b_*p**power, k=k, y0=sirs_dist[-1])[1:]))

plot_sir(sirs, sirs_dist, bprds[0], title='Late social distancing', save_as='ode_varparm_social_distancing_late')



# drug developement and distribution
kprds = [15, 55, 30]
ks = [0.01, 0.1, 0.8]
sirs = ode_simulation(T=kprds[0], b=b*p**power, k=ks[0], i0=i0)
sirs_drug = copy.copy(sirs[-1])
sirs_drug = sirs_drug[np.newaxis, :]
for prd, k_ in zip(kprds[1:], ks[1:]):
    sirs = np.vstack((sirs, ode_simulation(T=prd, b=b*p**power, k=ks[0], y0=sirs[-1])[1:]))
    sirs_drug = np.vstack((sirs_drug, ode_simulation(T=prd, b=b*p**power, k=k_, y0=sirs_drug[-1])[1:]))

plot_sir(sirs, sirs_drug, kprds[0], title='Early drug developement and distribution', save_as='ode_varparm_drug_early')

kprds = [30, 40, 30]
ks = [0.01, 0.1, 0.8]
sirs = ode_simulation(T=kprds[0], b=b*p**power, k=ks[0], i0=i0)
sirs_drug = copy.copy(sirs[-1])
sirs_drug = sirs_drug[np.newaxis, :]
for prd, k_ in zip(kprds[1:], ks[1:]):
    sirs = np.vstack((sirs, ode_simulation(T=prd, b=b*p**power, k=ks[0], y0=sirs[-1])[1:]))
    sirs_drug = np.vstack((sirs_drug, ode_simulation(T=prd, b=b*p**power, k=k_, y0=sirs_drug[-1])[1:]))

plot_sir(sirs, sirs_drug, kprds[0], title='Late drug developement and distribution', save_as='ode_varparm_drug_late')

# virus mutation
pprds = [20, 80]
ps = [0.1, 0.8]
sirs = ode_simulation(T=pprds[0], b=b*ps[0]**power, k=k, i0=i0)
sirs_mut = copy.copy(sirs[-1])
sirs_mut = sirs_mut[np.newaxis, :]
for prd, p_ in zip(pprds[1:], ps[1:]):
    sirs = np.vstack((sirs, ode_simulation(T=prd, b=b*ps[0]**power, k=k, y0=sirs[-1])[1:]))
    sirs_mut = np.vstack((sirs_mut, ode_simulation(T=prd, b=b*p_**power, k=k, y0=sirs_mut[-1])[1:]))
plot_sir(sirs, sirs_mut, pprds[0], title='Early virus mutation', save_as='ode_varparm_mutation_early')

pprds = [80, 20]
ps = [0.1, 0.8]
sirs = ode_simulation(T=pprds[0], b=b*ps[0]**power, k=k, i0=i0)
sirs_mut = copy.copy(sirs[-1])
sirs_mut = sirs_mut[np.newaxis, :]
for prd, p_ in zip(pprds[1:], ps[1:]):
    sirs = np.vstack((sirs, ode_simulation(T=prd, b=b*ps[0]**power, k=k, y0=sirs[-1])[1:]))
    sirs_mut = np.vstack((sirs_mut, ode_simulation(T=prd, b=b*p_**power, k=k, y0=sirs_mut[-1])[1:]))
plot_sir(sirs, sirs_mut, pprds[0], title='Late virus mutation', save_as='ode_varparm_mutation_late')


# all together

bprds = [5, 25, 70]
bs = [10, 3, 1]
kprds = [15, 55, 30]
ks = [0.01, 0.1, 0.8]
pprds = [20, 80]
ps = [0.1, 0.8]


sirs = ode_simulation(T=T, b=b*p**power, k=k, i0=i0)
sirs_all = ode_simulation(T=1, b=bs[0]*ps[0]**power, k=ks[0], i0=i0)[0]
sirs_all = sirs_all[np.newaxis, :]
for b_,k_,p_ in zip(np.repeat(bs, bprds), np.repeat(ks, kprds), np.repeat(ps, pprds)):
    sirs_all = np.vstack((sirs_all, ode_simulation(T=1, b=b_*p_**power, k=k_, y0=sirs_all[-1])[1:]))

plot_sir(sirs, sirs_all, 0, title='Early effects', save_as='ode_varparm_all')
