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

#class FitRoutinesSingle_Test(unittest.TestCase):
#    @classmethod
#    def setUpClass(cls):
#        cls.r1 = [4, 10]
#        cls.corr = Correlators("./test_data/corr_test_real.txt")
#        cls.frange, cls.fshape = fr.calculate_ranges(cls.r1, cls.corr.shape)
#
#    def test_fit_shapes_ncorr1(self):
#        fitres = fr.fit_single(f1, [1.], self.corr, self.frange)
#        self.assertIsNotNone(fitres)

#class FitRoutinesMulti_Test(unittest.TestCase):
#    @classmethod
#    def setUpClass(cls):
#        cls.r1 = [4, 10]
#        cls.r2 = [4, 12]
#        fnames = ["./test_data/corr_test_mat_%d%d.txt" % (s,t) \
#            for s in range(3) for t in range(3)]
#        corr = Correlators(fnames)
#        corr.gevp(1)
#        cls.corr = corr
#        cls.frange1, cls.fshape1 = fr.calculate_ranges(cls.r1, cls.corr.shape)
#        cls.frange2, cls.fshape2 = fr.calculate_ranges([cls.r1, cls.r1, cls.r1], cls.corr.shape)
#        cls.frange3, cls.fshape3 = fr.calculate_ranges([cls.r1, cls.r2, cls.r1], cls.corr.shape)
#
#    def test_fit_1_range(self):
#        fitres = fr.fit_single(f1, [1.], self.corr, self.frange1)
#        self.assertIsNotNone(fitres)
#
#    def test_fit_1_range_multi(self):
#        fitres = fr.fit_single(f1, [1.], self.corr, self.frange2)
#        self.assertIsNotNone(fitres)
#
#    def test_fit_2_range(self):
#        fitres = fr.fit_single(f1, [1.], self.corr, self.frange3)
#        self.assertIsNotNone(fitres)

#class FitRoutinesComb_Test(unittest.TestCase):
#    @classmethod
#    def setUpClass(cls):
#        cls.r1 = [4, 10]
#        cls.r2 = [4, 12]
#        fnames = ["./test_data/corr_test_mat_%d%d.txt" % (s,t) \
#            for s in range(3) for t in range(3)]
#        corr = Correlators(fnames)
#        corr.gevp(1)
#        cls.corr1 = corr
#        cls.corr = Correlators("./test_data/corr_test_real.txt")
#
#        cls.frange1, cls.fshape1 = fr.calculate_ranges(cls.r1, cls.corr.shape)
#        cls.fitres1 = fr.fit_single(f1, [1.], cls.corr, cls.frange1)
#
#    def test_fit(self):
#        franges, fshape = fr.calculate_ranges(self.r2, self.corr1.shape, self.fshape1)
#        def fitfunc(p, t, o):
#            return p + o
#        fitter = fr.fit_comb(fitfunc, [1.], self.corr1, franges, fshape, self.fitres1)
#        fitres = next(fitter)
#        self.assertIsNotNone(fitres)

if __name__ == "__main__":
    unittest.main()

