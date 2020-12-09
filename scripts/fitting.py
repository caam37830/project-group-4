import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
import pandas as pd

df = pd.read_csv('../data/cases.csv')
df.iloc[200]

df = df[['submission_date','new_case', 'tot_cases']]

cases = df.groupby(['submission_date']).sum().reset_index()
cases.drop('submission_date', axis=1, inplace=True)

pop = 328e6
cases['removed_case'] = np.hstack(([0]*20, cases.new_case[:-20].values ))
cases['d_I'] = cases.new_case - cases.removed_case
cases['I'] = np.cumsum(cases.d_I)
cases['i'] = cases.I/pop
cases['s'] = 1 - cases.tot_cases/pop


plt.plot(cases.new_case)
plt.title('Daily new Covid19 cases of in US')
plt.xlabel('Day from Jan 22, 2020')
plt.savefig('../docs/figs/data_new_cases.png', dpi=300)
plt.show()


fig, ax1 = plt.subplots()
ax1.plot(cases.s,'C0-')
ax1.set_ylabel(r"Susceptible Fraction, $s(t)$")
ax2 = ax1.twinx()
ax2.plot(cases.i, 'C1')
ax2.set_ylabel(r"Infectious Fraction, $i(t)$")
plt.xlabel('Day from Jan 22, 2020')
plt.show()



# solve
def fun(t, y, p):
    "right hand side of ODE"
    b = p[0]
    return np.vstack([-b*y[0,:]*y[1,:], b*y[0,:]*y[1,:]-0.05*y[1,:]])

def bc(ya, yb, p):
    "boundary condition residual"
    return np.array([ya[0] - sa, yb[0] - sb, ya[1] - ia])


si = np.asarray(cases[['s', 'i']]).transpose()
si.shape

def ode_fit(ta, tb):
    """
    Fit s(t), i(t) from day ta to tb
    """
    global sa, sb, ia, ib
    sa = si[0, ta]
    sb = si[0, tb]
    ia = si[1, ta]
    ib = si[1, tb]

    prd = tb - ta + 1
    y0 = si[:, ta:tb+1]
    t = np.arange(prd)
    sol = solve_bvp(fun, bc, t, y0, p=[0.1], tol=1e-10)
    return sol


# Find good cutoffs for piece-wise fitting
win = 10 # look backward and forward days
start = 50 # ignore early zeros
cutoffs = [start]
for t in range(start,si.shape[1]-win):
    if np.all(si[1,t-win:t]<=si[1,t]) and np.all(si[1,t]>=si[1,t:t+win]):
        cutoffs.append(t)
    elif np.all(si[1,t-win:t]>=si[1,t]) and np.all(si[1,t]<=si[1,t:t+win]):
        cutoffs.append(t)
cutoffs.append(si.shape[1]-1)

# 94, 142, 191 239

fig, (ax1, axb) = plt.subplots(2,1,figsize=(7,7), gridspec_kw={
                           'height_ratios': [4, 1]},sharex='all')
ax1.plot(si[0],'C0-')
ax1.set_ylabel(r"Susceptible Fraction$")
ax2 = ax1.twinx()
ax2.plot(si[1], 'C1')
ax2.set_ylabel(r"Infectious Fraction$")
plt.xlabel('Day from Jan 22, 2020')

bs = []
for i, ta in enumerate(cutoffs[:-1]):
    tb = cutoffs[i+1]
    sol = ode_fit(ta, tb)
    bs.append(sol.p[0])
    ax1.plot(np.linspace(ta, tb, sol.y.shape[1]), sol.y[0], 'C0--')
    ax2.plot(np.linspace(ta, tb, sol.y.shape[1]), sol.y[1], 'C1--')

ax1.plot([], [], 'C0-', label=r'$s(t)$', )
ax1.plot([], [], 'C1-', label=r'$i(t)$', )
ax1.plot([], [], 'C0--', label=r'$\hat s(t)$', )
ax1.plot([], [], 'C1--', label=r'$\hat i(t)$', )
ax1.legend(loc='center left')
ax1.set_title('True and Fitted Curves')

axb.set_title(r'Estimated Interaction Rate')
axb.set_ylabel(r'$\hat b$')
axb.set_xlim(0,si.shape[1])
axb.plot(range(cutoffs[0]+1, si.shape[1]), bs1,'grey')
axb.set_xlabel('Day from Jan 22, 2020')
plt.savefig('../docs/figs/ode_fitting.png', dpi=300)
plt.show()
