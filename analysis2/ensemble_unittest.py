"""
Unit tests for the ensemble class.
"""

import os
import unittest
import pickle
import numpy as np

from ensemble import LatticeEnsemble

class EnsBase_Test(unittest.TestCase):
    def setUp(self):
        self.dic = {"name": "test_dict", "L": 24, "T": 48, "T2": 25}
        self.ens = LatticeEnsemble(self.dic["name"], self.dic["L"], self.dic["T"])

    def test_init(self):
        self.assertEqual(self.dic, self.ens.data)

    def test_save(self):
        fname = "./test_data/test_dict.pkl"
        self.ens.save(fname)
        self.assertTrue(os.path.isfile(fname))
        with open(fname, "rb") as f:
            data = pickle.load(f)
        self.assertEqual(self.ens.data, data)

    def test_read(self):
        fname = "./test_data/test_dict.pkl"
        self.ens.save(fname)
        ens1 = LatticeEnsemble.read(fname)
        self.assertEqual(self.ens.data, ens1.data)

    def test_name(self):
        self.assertEqual(self.ens.name(), self.dic["name"])

    def test_T(self):
        self.assertEqual(self.ens.T(), self.dic["T"])

    def test_T2(self):
        self.assertEqual(self.ens.T2(), self.dic["T2"])

    def test_L(self):
        self.assertEqual(self.ens.L(), self.dic["L"])

    def test_add_data(self):
        newdata = {"test": 10.}
        self.ens.add_data("test", 10.)
        tmp = self.extractDictAFromB(newdata, self.ens.data)
        self.assertEqual(newdata, tmp)

    def test_get_data(self):
        newdata = {"test": 10.}
        self.ens.add_data("test", 10.)
        tmp = self.ens.get_data("test")
        self.assertEqual(tmp, newdata["test"])

    def extractDictAFromB(self,A,B):
        return dict([(k,B[k]) for k in A.keys() if k in B.keys()])

if __name__ == "__main__":
    unittest.main()
