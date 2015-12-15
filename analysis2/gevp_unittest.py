"""
Unit tests for the gevp functions.
"""

import unittest
import numpy as np

import gevp

class GEVPBase_Test(unittest.TestCase):
    def setUp(self):
        self.T=25
        self.mpi=0.14463
        self.Epipi0=0.306
        self.Epipi1=0.464
        self.time=np.linspace(0., self.T, self.T, endpoint=False)

    def test_shift1(self):
        data = np.vstack((5*np.exp(-self.Epipi0*self.time)+np.exp(-self.mpi*self.T),
                          4*np.exp(-self.Epipi0*self.time)+np.exp(-self.mpi*self.T)))
        sdata = gevp.gevp_shift_1(data, 1)
        tmp = (1. - np.exp(-self.Epipi0))
        cdata = 5. * tmp * np.exp(-self.Epipi0*self.time[:-1])
        self.assertTrue(np.allclose(sdata[0], cdata))
        cdata = 4. * tmp * np.exp(-self.Epipi0*self.time[:-1])
        self.assertTrue(np.allclose(sdata[1], cdata))

    def test_shift1_dE(self):
        data = np.vstack((5*np.exp(-self.Epipi1*self.time)+np.exp(-self.mpi*self.time),
                          4*np.exp(-self.Epipi1*self.time)+np.exp(-self.mpi*self.time)))
        sdata = gevp.gevp_shift_1(data, 1)
        # check that data is not compatible with just the normal exponential
        cdata = sdata[0,0] * np.exp(-self.Epipi1*self.time[:-1])
        self.assertFalse(np.allclose(sdata[0], cdata))
        cdata = sdata[0,1] * np.exp(-self.Epipi1*self.time[:-1])
        self.assertFalse(np.allclose(sdata[1], cdata))

        # do weight and shift
        dE = np.asarray([self.mpi, self.mpi])
        sdata = gevp.gevp_shift_1(data, 1, dE=dE)
        # compare data
        tmp = (1 - np.exp(-self.Epipi1)*np.exp(self.mpi))
        cdata = 5. * tmp * np.exp(-self.Epipi1*self.time[:-1])
        #print(sdata[0])
        #print(cdata)
        #print(sdata[0] - cdata)
        #print((sdata[0] - cdata)/cdata)
        self.assertTrue(np.allclose(sdata[0], cdata))
        cdata = 4. * tmp * np.exp(-self.Epipi1*self.time[:-1])
        #print(sdata[1] - cdata)
        self.assertTrue(np.allclose(sdata[1], cdata))

    def test_shift1_dE_cosh(self):
        data = np.vstack((5*np.exp(-self.Epipi1*self.time)+np.exp(-self.mpi*self.time),
                          4*np.exp(-self.Epipi1*self.time)+np.exp(-self.mpi*self.time)))
        sdata = gevp.gevp_shift_1(data, 1)
        # check that data is not compatible with just the normal exponential
        cdata = sdata[0,0] * np.exp(-self.Epipi1*self.time[:-1])
        self.assertFalse(np.allclose(sdata[0], cdata))
        cdata = sdata[0,1] * np.exp(-self.Epipi1*self.time[:-1])
        self.assertFalse(np.allclose(sdata[1], cdata))

        # do weight and shift
        dE = np.asarray([self.mpi, self.mpi])
        sdata = gevp.gevp_shift_1(data, 1, dE=dE)
        # compare data
        tmp = (1 - np.exp(-self.Epipi1)*np.exp(self.mpi))
        cdata = 5. * tmp * np.exp(-self.Epipi1*self.time[:-1])
        #print(sdata[0])
        #print(cdata)
        #print(sdata[0] - cdata)
        #print((sdata[0] - cdata)/cdata)
        self.assertTrue(np.allclose(sdata[0], cdata))
        cdata = 4. * tmp * np.exp(-self.Epipi1*self.time[:-1])
        #print(sdata[1] - cdata)
        self.assertTrue(np.allclose(sdata[1], cdata))
if __name__ == "__main__":
    unittest.main()
