"""
Unit tests for the fit routines.
"""

import unittest
import numpy as np

import fit_routines as fr

class FitRoutines_Test(unittest.TestCase):
    def test_fit_ranges(self):
        r1 = [4,10]
        r2 = [5,11]
        r3 = [6,12]
        s1 = (100, 25, 1)
        s2 = (100, 25, 2)
        s3 = (100, 25, 5)
        s4 = (100, 25, 9)
        tmp = fr.calculate_ranges([r1, r2], s2)
        self.assertEqual(tmp.shape, (2, 3, 2))
        tmp = fr.calculate_ranges(r1, s1)
        self.assertEqual(tmp.shape, (1, 3, 2))
        tmp1 = fr.calculate_ranges(r2, s3, tmp)
        self.assertEqual(tmp1.shape, (1, 3, 5, 3, 2))
        tmp2 = fr.calculate_ranges(r3, s4, tmp1)
        self.assertEqual(tmp2.shape, (1, 3, 5, 3, 9, 3, 2))

if __name__ == "__main__":
    unittest.main()

