"""
Implement tests for the two models
"""


import unittest
import sys
import os
sys.path.append('../')
from sir.agent import Person
from sir.agent_pop import Population
from sir.agent_seir import PopulationSEIR
from sir.agent_pop2d import Population2d
from sir.agent_varparm import PopulationVarparm

from sir.ode import ode_model
from sir.ode_varparm import ode_simulation as ode_simulation_varparm
from sir.ode_seir import ode_simulation as ode_simulation_seir
from sir.pde import pde_model

class Test_models(unittest.TestCase):

    def test_agent(self):
        p = Person()
        self.assertTrue(p.S)
        self.assertFalse(p.I)
        self.assertFalse(p.R)
        p.infect()
        self.assertFalse(p.S)
        self.assertTrue(p.I)
        self.assertFalse(p.R)
        p.recover()
        self.assertFalse(p.S)
        self.assertFalse(p.I)
        self.assertTrue(p.R)

    def test_agent_pop(self):
        pop = Population(N=1000, i0=0.01)
        result = pop.simulation(T=1, b=10, k=0.5)
        self.assertEqual(pop.count_sir().sum(),1)

        pop = Population(N=1000, i0=0.01)
        result = pop.simulation(T=0, b=0, k=0)
        self.assertEqual(pop.count_sir().sum(),1)

        pop = Population(N=1000, i0=0.01)
        result = pop.simulation(T=1, b=10, k=1)
        self.assertEqual(pop.count_sir().sum(),1)

    def test_agent_seir(self):
        pop = PopulationSEIR(N=1000, e0=0.1, i0=0.01)
        result = pop.simulation(T=1, b=10, f=0.5, k=0.5)
        self.assertEqual(pop.count_seir().sum(),1)

        pop = PopulationSEIR(N=1000, e0=0.1, i0=0.01)
        result = pop.simulation(T=0, b=0, f=0, k=0)
        self.assertEqual(pop.count_seir().sum(),1)

        pop = PopulationSEIR(N=1000, e0=0.1, i0=0.01)
        result = pop.simulation(T=1, b=10, f=1, k=1)
        self.assertEqual(pop.count_seir().sum(),1)

    def test_agent_pop2d(self):
        pop = Population2d(N=1000, I=10)
        result = pop.simulation(T=1, p=0.5, q=0.5, k=0.5)
        self.assertEqual(pop.count_SIR().sum(),1000)

        pop = Population2d(N=1000, I=10)
        result = pop.simulation(T=0, p=0, q=0, k=0)
        self.assertEqual(pop.count_SIR().sum(),1000)

        pop = Population2d(N=1000, I=1000)
        result = pop.simulation(T=1, p=2, q=1, k=1)
        self.assertEqual(pop.count_SIR().sum(),1000)

    def test_agent_varparm(self):
        pop = PopulationVarparm(N=1000, i0=0.01)
        result = pop.simulation(T=10,
                            b=[10]*10 + [5]*10 + [1]*80,
                            k=[0.01]*50 + [0.1]*30 + [0.5]*20,
                            p=[0.1]*20 + [0.8]*80)
        self.assertEqual(pop.count_sir().sum(),1)

    def test_ode(self):
        i0 = 0.1/100
        T = 100
        bs = [1,5,10]
        ks = [0.01, 0.1,1]
        for b in bs:
            for k in ks:
                s,i,r = ode_model(i0, T, b, k)
                for t in range(len(s)): # every day
                    self.assertAlmostEqual(s[t]+i[t]+r[t],1, msg= f'failed when t={t}, b={b}, k={k}: s+i+r={s[t]+i[t]+r[t]}')

    def test_ode_varparm(self):
        sirs = ode_simulation_varparm(T=10, b=10, k=0.1, i0=0.001)
        sirs = ode_simulation_varparm(T=10, b=0, k=0, i0=0.001)
        sirs = ode_simulation_varparm(T=10, b=1, k=1, i0=0.001)

        sirs = ode_simulation_varparm(T=10, b=10, k=0.1, y0=[0,0,1])
        sirs = ode_simulation_varparm(T=10, b=10, k=0.1, y0=[0,1,0])
        sirs = ode_simulation_varparm(T=10, b=10, k=0.1, y0=[1,0,0])

    def test_ode_seir(self):
        seirs = ode_simulation_seir(e0=0.001, i0=0.001, T=10, b=10, f=0.1, k=0.1)
        seirs = ode_simulation_seir(e0=0.001, i0=0.001, T=10, b=10, f=0, k=0)
        seirs = ode_simulation_seir(e0=0.001, i0=0.001, T=10, b=10, f=1, k=1)

    def test_pde(self):
        sol = pde_model(i0=0.001, pos=(100,100), M=200, T=10, b=10, k=0.5, p=0.0001)
        sol = pde_model(i0=0.001, pos=(0,0), M=200, T=10, b=10, k=0, p=0)
        sol = pde_model(i0=0.001, pos=(100,100), M=200, T=10, b=10, k=1, p=0.0001)
