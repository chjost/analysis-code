"""
Unit tests for the fit routines.
"""

import unittest
import numpy as np

import fit_routines as fr
from correlator import Correlators
from functions import func_const as f1

depth = lambda L: isinstance(L, list) and max(map(depth, L))+1

class FitRoutinesBase_Test(unittest.TestCase):
    def test_range_ncorr1(self):
        r1 = [4,10]
        s1 = (100, 25, 1)
        ranges, shape = fr.calculate_ranges(r1, s1)
        self.assertEqual(shape, [[3]])
        self.assertIsInstance(ranges, list)
        self.assertEqual(s1[-1], len(ranges))

    def test_range_ncorr2(self):
        r1 = [4,10]
        r2 = [4,12]
        s1 = (100, 25, 2)
        ranges, shape = fr.calculate_ranges([r1, r2], s1)
        self.assertEqual(shape, [[3, 6]])
        self.assertNotEqual(shape, [3, 6])
        self.assertIsInstance(ranges, list)
        self.assertEqual(s1[-1], len(ranges))

    def test_range_ncorr10(self):
        r1 = [4,10]
        s1 = (100, 25, 10)
        ranges, shape = fr.calculate_ranges(r1, s1)
        self.assertEqual(shape, [[3]*10])
        self.assertIsInstance(ranges, list)
        self.assertEqual(s1[-1], len(ranges))

    def test_range_seq_ncorr1(self):
        r1 = [4,10]
        s1 = (100, 25, 1)
        ranges, shape = fr.calculate_ranges(r1, s1)
        self.assertEqual(shape, [[3]])
        self.assertIsInstance(ranges, list)
        self.assertEqual(s1[-1], len(ranges))
        ranges, shape = fr.calculate_ranges(r1, s1, oldshape=shape)
        self.assertEqual(shape, [[3], [3]])
        self.assertIsInstance(ranges, list)
        self.assertEqual(s1[-1], len(ranges))

    def test_range_seq_ncorr2(self):
        r1 = [4,10]
        r2 = [4,12]
        s1 = (100, 25, 2)
        ranges, shape = fr.calculate_ranges([r1, r2], s1)
        self.assertEqual(shape, [[3, 6]])
        self.assertIsInstance(ranges, list)
        self.assertEqual(s1[-1], len(ranges))
        ranges, shape = fr.calculate_ranges([r1, r2], s1, oldshape=shape)
        self.assertEqual(shape, [[3, 6], [3, 6]])
        self.assertIsInstance(ranges, list)
        self.assertEqual(s1[-1], len(ranges))

    def test_range_seq_ncorr1_2(self):
        r1 = [4,10]
        r2 = [4,12]
        s1 = (100, 25, 1)
        s2 = (100, 25, 2)
        ranges, shape = fr.calculate_ranges(r1, s1)
        self.assertEqual(shape, [[3]])
        self.assertIsInstance(ranges, list)
        self.assertEqual(s1[-1], len(ranges))
        ranges, shape = fr.calculate_ranges(r2, s2, oldshape=shape)
        self.assertEqual(shape, [[3], [6, 6]])
        self.assertIsInstance(ranges, list)
        self.assertEqual(s2[-1], len(ranges))

class FitRoutines_Test(unittest.TestCase):
    def setUp(self):
        self.r1 = [4, 10]
        self.r2 = [4, 12]
        self.corr = Correlators("./test_data/corr_test_real.txt")
        fnames = ["./test_data/corr_test_mat_%d%d.txt" % (s,t) \
            for s in range(3) for t in range(3)]
        self.corr1 = Correlators(fnames)
        self.corr1.gevp(1)

    def test_fit_shapes_ncorr1(self):
        fit, chi, res= fr.fit(f1, [1.], self.corr, self.r1)
        print(fit)
        self.assertEqual(len(fit), 1)
        self.assertEqual(fit, [(404, 1, 3)])
        self.assertEqual(len(chi), 1)
        self.assertEqual(chi, [(404, 3)])

    def test_fit_shapes_ncorr2(self):
        fit, chi, res = fr.fit(f1, [1.], self.corr1, self.r1)
        print(fit)
        self.assertEqual(len(fit), 3)
        self.assertEqual(fit, [(404, 1, 3),] * 3)
        self.assertEqual(len(chi), 3)
        self.assertEqual(chi, [(404, 3),] * 3)

if __name__ == "__main__":
    unittest.main()

