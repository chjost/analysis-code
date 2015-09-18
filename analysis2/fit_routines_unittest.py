"""
Unit tests for the fit routines.
"""

import unittest
import numpy as np

import fit_routines as fr

class FitRoutines_Test(unittest.TestCase):
    def test_fit_ranges(self):
        tmp = fr.calculate_ranges([4,10], (100, 25, 1))
        print(tmp.shape)
        print(" ")
        tmp = fr.calculate_ranges([4,10], (100, 25, 2))
        print(tmp.shape)
        print(" ")
        tmp = fr.calculate_ranges([[4,10], [5, 11]], (100, 25, 2))
        print(tmp.shape)
        print(" ")
        tmp = fr.calculate_ranges([4,10], (100, 25, 1))
        print(tmp.shape)
        print(" ")
        tmp1 = fr.calculate_ranges([5,11], (100, 25, 5), tmp)
        print(tmp1.shape)
        print(" ")
        tmp2 = fr.calculate_ranges([6,12], (100, 25, 9), tmp1)
        print(tmp2.shape)

if __name__ == "__main__":
    unittest.main()

