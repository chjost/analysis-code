#!/usr/bin/python
import unittest
import numpy as np
import pandas as pd
#import boot_statistics
from boot import *

class TestBootMethods(unittest.TestCase):

    def test_pseudo_strapping(self):
	np.random.seed(1227)
        mean,std,shape = 5.,0.2,(15,)
        data = std * np.random.randn(*shape) + mean 
	data[0] = mean
	fix = pd.Series(data)
        boot = Boot()
	trial = boot.strap(method='parametric',data=np.array((mean,std)),
                           params={'size':shape,'seed':1227})
        self.assertEqual(trial.any(),fix.any())

if __name__ == '__main__':
	unittest.main()

