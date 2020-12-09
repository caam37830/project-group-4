"""
Implement tests for the two models
"""


import unittest
import sys
import os
sys.path.append('../')
from sir.agent import Person
from sir.ode import ode_model



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
