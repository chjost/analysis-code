"""
Unit tests for the correlator class.
"""
import unittest
import numpy as np

from correlator import Correlators

# Testing the initialization of the Correlators class
class CorBase_Test(unittest.TestCase):
    def setUp(self):
        # read in data to check against
        fname = "./test_data/corr_test_real_short.npy"
        self.data = np.atleast_3d(np.load(fname))
        fname = "./test_data/corr_test_mat_short_sym.npy"
        self.mat = np.load(fname)
    
    def test_read_nodir(self):
        # test reading if directory does not exist
        fname = "./nodir/data.txt"
        self.assertRaises(IOError, Correlators, fname)

    def test_read_nofile(self):
        # test reading if file does not exist
        fname = "./test_data/nodata.txt"
        self.assertRaises(IOError, Correlators, fname)

    def test_read_data(self):
        # test the reading of a file
        fname = "./test_data/corr_test_real_short.txt"
        corr = Correlators(fname)
        self.assertTrue(np.all(corr.data == np.atleast_3d(self.data[:,:,1])))

        # read data with comments
        fname = "./test_data/corr_test_real_comments.txt"
        corr = Correlators(fname)
        self.assertTrue(np.all(corr.data == np.atleast_3d(self.data[:,:,1])))

    def test_arg_column(self):
        # file has 3 columns, the second column is read by default
        # read colums 0 and 1
        fname = "./test_data/corr_test_real_short.txt"
        corr = Correlators(fname, column=(0,1))
        self.assertTrue(np.all(corr.data == self.data[:,:,:2]))

        # read non-existing column
        self.assertRaises(ValueError, Correlators, fname, column=(3,))

    def test_arg_skip(self):
        # skip is 1 by default
        # set skip < 1
        fname = "./test_data/corr_test_real_short.txt"
        self.assertRaises(ValueError, Correlators, fname, skip=0)

        # skip bigger header
        fname = "./test_data/corr_test_real_header.txt"
        corr = Correlators(fname, skip=5)
        self.assertTrue(np.all(corr.data == np.atleast_3d(self.data[:,:,1])))

    def test_read_matrix(self):
        fnames = ["./test_data/corr_test_mat_short_%d%d.txt" % (s,t) \
            for s in range(3) for t in range(3)]
        corr = Correlators(fnames)
        self.assertEqual(self.mat[:,:,1].shape, corr.shape)
        self.assertTrue(np.allclose(self.mat[:,:,1], corr.data))

    def test_read_matrix_columns(self):
        # the file has 3 columns, 2nd column read by default
        fnames = ["./test_data/corr_test_mat_short_%d%d.txt" % (s,t) \
            for s in range(3) for t in range(3)]
        corr = Correlators(fnames, column=(0,1))
        self.assertEqual(self.mat[:,:,:2].shape, corr.shape)
        self.assertTrue(np.allclose(self.mat[:,:,:2], corr.data))

# Testing the data handling with one correlation function
class CorrFunc_Test(unittest.TestCase):
    def setUp(self):
        # read in data
        fname = "./test_data/corr_test_real.txt"
        self.corr = Correlators(fname)

    def test_write_data(self):
        fname = "./test_data/tmp_data.npy"
        self.corr.save(fname)
        corr1 = Correlators.read(fname)
        self.assertTrue(np.allclose(corr1.data, self.corr.data))

    def test_shape(self):
        data = np.load("./test_data/corr_test_real.npy")
        self.assertEqual(self.corr.shape, data.shape + (1,))

    def test_symmetrize(self):
        self.corr.symmetrize()
        self.assertEqual(self.corr.shape, (404, 25, 1))
        # TODO: get data to check against
        #self.assertAlmostEqual()

    def test_bootstrap(self):
        self.corr.bootstrap(100)
        self.assertEqual(self.corr.shape, (100, 48, 1))
        # TODO: get data to check against
        #self.assertAlmostEqual()

    def test_sym_and_boot(self):
        self.corr.sym_and_boot(100)
        self.assertEqual(self.corr.shape, (100, 25, 1))
        # TODO: get data to check against
        #self.assertAlmostEqual()

    def test_shift_1(self):
        # the shift function should not do anything
        self.corr.symmetrize()
        before = np.copy(self.corr.data)
        self.corr.shift(1)
        self.assertEqual(self.corr.shape, before.shape)
        self.assertTrue(np.array_equal(before, self.corr.data))

    def test_shift_1_weight(self):
        self.corr.symmetrize()
        before = np.copy(self.corr.data)
        self.corr.shift(1, 1.)
        self.assertEqual(self.corr.shape, before.shape)
        self.assertTrue(np.array_equal(before, self.corr.data))

    def test_shift_2(self):
        self.corr.symmetrize()
        before = np.copy(self.corr.data)
        self.corr.shift(1, 1., shift=2)
        self.assertEqual(self.corr.shape, before.shape)
        self.assertTrue(np.array_equal(before, self.corr.data))
        # usually this would throw an exception, but in this case
        # nothing should happen
        self.corr.shift(1, shift=2)
        self.assertEqual(self.corr.shape, before.shape)
        self.assertTrue(np.array_equal(before, self.corr.data))

    def test_mass_acosh(self):
        self.corr.symmetrize()
        self.corr.mass()
        self.assertEqual(self.corr.shape, (404, 23, 1))
        # TODO: get data to check against
        #self.assertAlmostEqual()

    def test_mass_log(self):
        self.corr.symmetrize()
        self.corr.mass(False)
        self.assertEqual(self.corr.shape, (404, 24, 1))
        # TODO: get data to check against
        #self.assertAlmostEqual()

# Testing the data handling as a correlation function matrix
class CorrMatrix_test(unittest.TestCase):
    def setUp(self):
        # read in data
        fnames = ["./test_data/corr_test_mat_%d%d.txt" % (s,t) \
            for s in range(3) for t in range(3)]
        self.corr = Correlators(fnames)

    def test_write_data(self):
        fname = "./test_data/tmp_data.npy"
        self.corr.save(fname)
        corr1 = Correlators.read(fname)
        self.assertTrue(np.allclose(corr1.data, self.corr.data))

    def test_shape(self):
        data = np.load("./test_data/corr_test_mat_sym.npy")
        self.assertEqual(self.corr.shape, data.shape)

    def test_symmetrize(self):
        self.corr.symmetrize()
        self.assertEqual(self.corr.shape, (404, 25, 3, 3))
        # TODO: get data to check against
        #self.assertAlmostEqual()

    def test_bootstrap(self):
        self.corr.bootstrap(100)
        self.assertEqual(self.corr.shape, (100, 48, 3, 3))
        # TODO: get data to check against
        #self.assertAlmostEqual()

    def test_sym_and_boot(self):
        self.corr.sym_and_boot(100)
        self.assertEqual(self.corr.shape, (100, 25, 3, 3))
        # TODO: get data to check against
        #self.assertAlmostEqual()

    def test_shift_1(self):
        self.corr.symmetrize()
        self.corr.shift(1)
        self.assertEqual(self.corr.shape, (404, 24, 3, 3))
        # TODO: get data to check against
        #self.assertAlmostEqual()

    def test_shift_1_weight(self):
        self.corr.symmetrize()
        self.corr.shift(1, 1.)
        self.assertEqual(self.corr.shape, (404, 24, 3, 3))
        #self.assertTrue(np.all())
        # TODO: get data to check against
        #self.assertAlmostEqual()

    def test_shift_2(self):
        self.corr.symmetrize()
        self.corr.shift(1, 1., shift=2)
        self.assertEqual(self.corr.shape, (404, 24, 3, 3))
        # TODO: get data to check against
        #self.assertAlmostEqual()

    def test_shift_2_noweight(self):
        self.corr.symmetrize()
        self.assertRaises(ValueError, self.corr.shift, 1, shift=2)

    def test_mass_acosh(self):
        self.corr.symmetrize()
        self.corr.mass()
        self.assertEqual(self.corr.shape, (404, 23, 3, 3))
        # TODO: get data to check against
        #self.assertAlmostEqual()

    def test_mass_log(self):
        self.corr.symmetrize()
        self.corr.mass(False)
        self.assertEqual(self.corr.shape, (404, 24, 3, 3))
        # TODO: get data to check against
        #self.assertAlmostEqual()

if __name__ == "__main__":
    unittest.main()

