"""
Unit tests for different functions.
"""

import unittest
import numpy as np

import functions as func

class Func_Test(unittest.TestCase):
    def test_derivative(self):
        data = np.arange(10)
        self.assertRaises(IndexError, func.compute_derivative, data)
        data = np.arange(20).reshape((2,10))
        data1 = func.compute_derivative(data)
        self.assertEqual(data1.shape, (2,9))
        self.assertTrue(np.array_equiv(data1, np.array([1.])))

    def test_eff_mass(self):
        pass

    def test_error(self):
        pass

    def test_single_corr(self):
        pass

    def test_ratio(self):
        pass

    def test_const(self):
        pass

if __name__ == "__main__":
    unittest.main()

