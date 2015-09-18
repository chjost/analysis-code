"""
Unit tests for the fit class.
"""

import unittest
import numpy as np

from fit import LatticeFit, FitResult

class Fit_Test(unittest.TestCase):
    def setUp(self):
        self.fitter = LatticeFit()

    def test_base(self):
        self.assertTrue(True)

    def test_fit(self):
        pass

class FitResult_Test(unittest.TestCase):
    def test_init(self):
        pass

    def test_add_data_single(self):
        fr = FitResult()
        data = np.ones((10, 25, 30))
        chi2 = np.ones_like(data)
        fr.add_data(data, chi2, ["single"])
        
    def test_add_data_multiple(self):
        fr = FitResult()
        d = np.ones((10, 25, 30))
        data = [d for i in range(5)]
        c = np.ones_like(d)
        chi2 = [c for i in range(5)]
        fr.add_data(data, chi2, ["multiple"])

    def test_get_empty(self):
        fr = FitResult()
        self.assertRaises(ValueError, fr.get_empty, (10, 25, 30), ["d"], [1,2])
        tmp = fr.get_empty((10, 25, 30), ["a", "b"], [1, 5])

    def test_get_fit_ranges(self):
        fr = FitResult()
        tmp = fr.get_ranges()
        self.assertIsNone(tmp)
        
    def test_save(self):
        pass

    def test_read(self):
        pass

if __name__ == "__main__":
    unittest.main()

