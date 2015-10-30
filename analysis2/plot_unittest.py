"""
Unit tests for the plot class.
"""

import unittest
import numpy as np

from plot import LatticePlot
from functions import func_single_corr as f2

class Fit_Test(unittest.TestCase):

    def test_plot_data(self):
        plotter = LatticePlot("./test_data/test_plot_data.pdf")
        # create the plot interval
        x = np.linspace(0., 24., 24)
        # some sample arguments, similar to the pion on A40.24
        args = [100., 0.14463]
        # time extent similar to A40.24
        add = [48.]
        # generate data
        y = f2(args, x, add)
        # error of 1%
        dy = y * 0.01
        # plot data
        plotter.plot_data(x, y, dy, "data")
        plotter.save()

    def test_plot_function(self):
        plotter = LatticePlot("./test_data/test_plot_function.pdf")
        # plot the function in plot interval
        x = np.linspace(0., 24., 1000)
        # some sample arguments, similar to the pion on A40.24
        args = [100., 0.14463]
        # time extent similar to A40.24
        add = [48.]
        # plot standard case
        plotter.plot_function(f2, x, args, "1arg, 1add", add)
        plotter.save()
        # add more data to additional arguments
        add = np.linspace(47., 49., 7)
        plotter.plot_function(f2, x, args, "1arg, 7add", add)
        plotter.save()
        # add more data to arguments
        args = np.vstack((np.linspace(95., 105., 7), np.linspace(0.144, 0.145, 7)))
        plotter.plot_function(f2, x, args.T, "7arg, 7add", add)
        plotter.save()
        # back to one additional argument
        add = [48.]
        plotter.plot_function(f2, x, args.T, "10arg, 1add", add)
        plotter.save()

if __name__ == "__main__":
    unittest.main()

