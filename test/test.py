"""
Implement tests for the two models
"""

import unittest
import sys
sys.path.append("..")
from sir import *



class Test_models(unittest.TestCase):
    
    def test_agent(self):
        p = Person()
        self.assertTrue(p.S)
        self.assertFalse(p.I)
        self.assertFalse(p.R)
        p.infec()
        self.assertFalse(p.S)
        self.assertTrue(p.I)
        self.assertFalse(p.R)
        p.recover()
        self.assertFalse(p.S)
        self.assertFalse(p.I)
        self.assertTrue(p.R)

    def test_ode(self):
        N = 100
        i0 = 0.1/100
        Ts = [1,10,100]
        bs = [1,5,10]
        ks = [0.01, 0.1,1]
        for T in Ts:
            for b in bs:
                for k in ks:
                    s,i,r = ode_model(N, i0, T, b, k)
                    self.assertTrue(s + i + r == 1)
