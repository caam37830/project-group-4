"""
Implement tests for the two models
"""

import unittest
import sys
sys.path.append("..")
from sir import *


class Test_models(unittest.TestCase):
    def test_agent(self):
        pass

    def test_ode(self):
        N = 1000
        i0 = 0.1/100
        t = 20
        b = 10
        k = 0.1

        s,i,r = ode_model(N, i0, t, b, k)
        self.assertTrue(s + i + r == 1)
