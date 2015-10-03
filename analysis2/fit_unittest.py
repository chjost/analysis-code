"""
Unit tests for the fit class.
"""

from __future__ import with_statement

import os
import unittest
import numpy as np

from fit import LatticeFit, FitResult
from functions import func_const as f1

class Fit_Test(unittest.TestCase):

    def test_fit(self):
        fitter = LatticeFit(f1)
        self.assertIsNotNone(fitter)

class FitResult_Test(unittest.TestCase):

    def test_add_data_single(self):
        fr = FitResult("")
        res = np.ones((10, 25))
        chi2 = np.ones((10,))
        pval = np.ones((10,))
        self.assertRaises(RuntimeError, fr.add_data, (1,), res, chi2, pval)
        fr.create_empty((10, 25, 4), (10, 4), 3)
        fr.add_data((0,0), res, chi2, pval)
        self.assertTrue(np.array_equal(fr.data[0][:,:,0], res))
        self.assertTrue(np.array_equal(fr.chi2[0][:,0], chi2))
        self.assertTrue(np.array_equal(fr.pval[0][:,0], pval))
        fr.add_data((2,0), res, chi2, pval)
        self.assertTrue(np.array_equal(fr.data[2][:,:,0], res))
        self.assertTrue(np.array_equal(fr.chi2[2][:,0], chi2))
        self.assertTrue(np.array_equal(fr.pval[2][:,0], pval))
        fr.add_data((0,3), res, chi2, pval)
        self.assertTrue(np.array_equal(fr.data[0][:,:,3], res))
        self.assertTrue(np.array_equal(fr.chi2[0][:,3], chi2))
        self.assertTrue(np.array_equal(fr.pval[0][:,3], pval))

    def test_get_index_len1(self):
        fr = FitResult("")
        self.assertRaises(RuntimeError, fr._get_index, (1,))
        fr.create_empty((10, 25, 4), (10, 4), 3)
        self.assertRaises(ValueError, fr._get_index, (1,2))
        self.assertEqual(fr._get_index(0), 0)
        self.assertEqual(fr._get_index(1), 1)
        self.assertEqual(fr._get_index(2), 2)
        self.assertRaises(ValueError, fr._get_index, (3,))

    def test_get_index_len2(self):
        fr = FitResult("")
        fr.create_empty((10, 25, 4), (10, 4), [2, 3])
        self.assertEqual(fr._get_index((0, 0)), 0)
        self.assertEqual(fr._get_index((0, 1)), 1)
        self.assertEqual(fr._get_index((0, 2)), 2)
        self.assertEqual(fr._get_index((1, 0)), 3)
        self.assertEqual(fr._get_index((1, 1)), 4)
        self.assertEqual(fr._get_index((1, 2)), 5)

    def test_get_index_len3(self):
        fr = FitResult("")
        fr.create_empty((10, 25, 4), (10, 4), [2, 3, 4])
        self.assertEqual(fr._get_index((0, 0, 0)), 0)
        self.assertEqual(fr._get_index((0, 0, 1)), 1)
        self.assertEqual(fr._get_index((0, 0, 2)), 2)
        self.assertEqual(fr._get_index((0, 0, 3)), 3)
        self.assertEqual(fr._get_index((0, 1, 0)), 4)
        self.assertEqual(fr._get_index((0, 1, 1)), 5)
        self.assertEqual(fr._get_index((0, 1, 2)), 6)
        self.assertEqual(fr._get_index((0, 1, 3)), 7)
        self.assertEqual(fr._get_index((0, 2, 0)), 8)
        self.assertEqual(fr._get_index((0, 2, 1)), 9)
        self.assertEqual(fr._get_index((0, 2, 2)), 10)
        self.assertEqual(fr._get_index((0, 2, 3)), 11)
        self.assertEqual(fr._get_index((1, 0, 0)), 12)
        self.assertEqual(fr._get_index((1, 0, 1)), 13)
        self.assertEqual(fr._get_index((1, 0, 2)), 14)
        self.assertEqual(fr._get_index((1, 0, 3)), 15)
        self.assertEqual(fr._get_index((1, 1, 0)), 16)
        self.assertEqual(fr._get_index((1, 1, 1)), 17)
        self.assertEqual(fr._get_index((1, 1, 2)), 18)
        self.assertEqual(fr._get_index((1, 1, 3)), 19)
        self.assertEqual(fr._get_index((1, 2, 0)), 20)
        self.assertEqual(fr._get_index((1, 2, 1)), 21)
        self.assertEqual(fr._get_index((1, 2, 2)), 22)
        self.assertEqual(fr._get_index((1, 2, 3)), 23)

    def test_create_empty(self):
        fr = FitResult("")
        self.assertRaises(ValueError, fr.create_empty, (10, 25, 4), (10, 25, 4), 1)
        self.assertRaises(RuntimeError, fr.create_empty, (10, 25, 4), (10, 25, 4), 1)
        fr = FitResult("")
        fr.create_empty((10, 25, 4), (10, 4), [2, 3])
        self.assertIsNotNone(fr.data)

    def test_get_ranges(self):
        fr = FitResult("")
        tmp = fr.get_ranges()
        self.assertEqual(tmp, (None, None))

    def test_set_ranges(self):
        fr = FitResult("")
        ran = np.ones((100, 25, 3))
        shape = (100, 25, 3)
        fr.set_ranges(ran, shape)
        self.assertTrue(np.array_equal(ran, fr.fit_ranges))
        self.assertTrue(np.array_equal(shape, fr.fit_ranges_shape))

    def test_save(self):
        fr = FitResult("")
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
                self.assertEqual(len(L), 6*4+2)
                tmp = f['pi00']
                self.assertTrue(np.array_equal(np.zeros((10, 25, 4)), tmp))

    def test_read(self):
        fr = FitResult("test")
        res = np.ones((10, 25))
        chi2 = np.ones((10,))
        pval = np.ones((10,))
        self.assertRaises(RuntimeError, fr.add_data, (1,), res, chi2, pval)
        fr.create_empty((10, 25, 4), (10, 4), 3)
        fr.add_data((0,0), res, chi2, pval)
        fr.add_data((2,0), res, chi2, pval)
        fr.add_data((0,3), res, chi2, pval)
        fname = "./test_data/tmp_fitresult.npz"
        fr.save(fname)
        fexists = os.path.isfile(fname)
        if fexists:
            fr1 = FitResult.read(fname)
            self.assertTrue(np.array_equal(fr1.data[0][:,:,0], res))
            self.assertTrue(np.array_equal(fr1.chi2[0][:,0], chi2))
            self.assertTrue(np.array_equal(fr1.pval[0][:,0], pval))
            self.assertTrue(np.array_equal(fr1.data[2][:,:,0], res))
            self.assertTrue(np.array_equal(fr1.chi2[2][:,0], chi2))
            self.assertTrue(np.array_equal(fr1.pval[2][:,0], pval))
            self.assertTrue(np.array_equal(fr1.data[0][:,:,3], res))
            self.assertTrue(np.array_equal(fr1.chi2[0][:,3], chi2))
            self.assertTrue(np.array_equal(fr1.pval[0][:,3], pval))
            self.assertEqual(fr1.corr_id, "test")
    
    def test_get_data(self):
        fr = FitResult("")
        res = np.ones((10, 25))
        chi2 = np.ones((10,))
        pval = np.ones((10,))
        self.assertRaises(RuntimeError, fr.add_data, (1,), res, chi2, pval)
        fr.create_empty((10, 25, 4), (10, 4), 3)
        fr.add_data((0,0), res, chi2, pval)
        self.assertTrue(np.array_equal(fr.data[0][:,:,0], np.ones((10, 25))))
        self.assertTrue(np.array_equal(fr.chi2[0][:,0], np.ones((10,))))
        fr.add_data((2,0), res, chi2, pval)
        self.assertTrue(np.array_equal(fr.data[2][:,:,0], np.ones((10, 25))))
        self.assertTrue(np.array_equal(fr.chi2[2][:,0], np.ones((10,))))
        fr.add_data((0,3), res, chi2, pval)
        self.assertTrue(np.array_equal(fr.data[0][:,:,3], np.ones((10, 25))))
        self.assertTrue(np.array_equal(fr.chi2[0][:,3], np.ones((10,))))

if __name__ == "__main__":
    unittest.main()

