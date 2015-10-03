"""
Unit tests for the plot class.
"""

import unittest
import numpy as np

from plot import LatticePlot
from functions import func_single_corr as f2

class Fit_Test(unittest.TestCase):
    def test_base(self):
        self.assertTrue(True)

    def test_save(self):
        self.assertTrue(True)

    def test_plot_function(self):
        plotter = LatticePlot("./test.pdf")
        x = np.linspace(0., 24., 1000)
        args = [-0.00184, 0.48139]
        add = [25.]
        plotter.plot_function(f2, x, args, add, [10, 16])
        plotter.save()

if __name__ == "__main__":
    unittest.main()

