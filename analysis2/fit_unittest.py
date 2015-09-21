"""
Unit tests for the fit class.
"""

import os
import unittest
import numpy as np

import fit
from functions import func_const as f1

class Fit_Test(unittest.TestCase):
    def setUp(self):
        self.fitter = fit.LatticeFit(f1)

    def test_base(self):
        self.assertTrue(True)

    def test_fit(self):
        pass

class FitResult_Test(unittest.TestCase):

    def test_add_data(self):
        pass

    def test_create_empty(self):
        fr = fit.FitResult()
        self.assertRaises(ValueError, fr.create_empty, (10, 25, 4), (10, 25, 4), 1)
        fr.create_empty((10, 25, 4), (10, 4), [2, 3])
        self.assertIsNotNone(fr.data)

    def test_get_ranges(self):
        fr = fit.FitResult()
        tmp = fr.get_ranges()
        self.assertEqual(tmp, (None, None))

    def test_set_ranges(self):
        fr = fit.FitResult()
        ran = np.ones((100, 25, 3))
        shape = (100, 25, 3)
        fr.set_ranges(ran, shape)
        self.assertTrue(np.array_equal(ran, fr.fit_ranges))
        self.assertTrue(np.array_equal(shape, fr.fit_ranges_shape))

    def test_save(self):
        fr = fit.FitResult()
        ran = np.ones((100, 25, 3))
        shape = (100, 25, 3)
        fr.set_ranges(ran, shape)
        fr.create_empty((10, 25, 4), (10, 4), [2, 3])
        fname = "./test_data/tmp_fitresult.npz"
        fr.save(fname)
        fexists = os.path.isfile(fname)
        self.assertTrue(fexists)
        if fexists:
            with np.load(fname) as f:
                L = f.files
                self.assertEqual(len(L), 6*4+1)
                tmp = f['pi00']
                print(type(tmp))
                self.assertTrue(np.array_equal(np.zeros((10, 25, 4)), tmp))

    def test_read(self):
        pass

if __name__ == "__main__":
    unittest.main()

