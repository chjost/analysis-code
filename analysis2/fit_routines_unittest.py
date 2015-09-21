"""
Unit tests for the fit routines.
"""

import unittest
import numpy as np

import fit_routines as fr

depth = lambda L: isinstance(L, list) and max(map(depth, L))+1

class FitRoutines_Test(unittest.TestCase):
    def test_fit_ranges(self):
        r1 = [4,10]
        r2 = [4,12]
        r3 = [4,14]
        s1 = (100, 25, 1)
        s2 = (100, 25, 2)
        s3 = (100, 25, 5)
        s4 = (100, 25, 9)
        tmp, shape= fr.calculate_ranges([r1, r2], s2)
        self.assertEqual(shape, [[3, 6]])
        self.assertEqual(depth(tmp), 2)
        tmp1, shape = fr.calculate_ranges(r2, s3, shape)
        self.assertEqual(shape, [[3,6], [6]*s3[-1]])
        self.assertEqual(depth(tmp1), 4)
        tmp2, shape = fr.calculate_ranges(r3, s4, shape)
        #self.assertEqual(shape, [[3,6], [6]*s3[-1], [10]*s4[-1]])
        #self.assertEqual(depth(tmp2), 6)

if __name__ == "__main__":
    unittest.main()

