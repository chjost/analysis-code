import numpy as np

def pion_fit(p, t, o):
    return 0.5*p[0]*p[0]*(np.exp(-p[1]*t)+np.exp(-p[1]*(o-t)))
