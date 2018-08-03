import numpy as np

def get_beta_name(b):
    if b == 1.90:
        return 'A'
    elif b == 1.95:
        return 'B'
    elif b == 2.10:
        return 'D'
    else:
        print('bet not known')

def get_mul_name(l):
    return int(l*10**4)

def get_mus_name(s):
    if s in [0.0115,0.013,0.0185,0.016]:
        return 'lo'
    elif s in [0.015,0.0186,0.0225]:
        return 'mi'
    elif s in [0.018,0.021,0.02464]:
        return 'hi'
    else:
        print('mu_s not known')

def ensemblenames(ix_values):
    """convert index tuples (beta, L, mu_l) to ensemblenames
    """
    ensemblelist = []
    for i,e in enumerate(ix_values):
        b = get_beta_name(e[0])
        l=int(e[1])
        mul = get_mul_name(e[2])
        #string = '%s%d %s'%(b,mul,mus)
        string = '%s%d.%d'%(b,mul,l)
        ensemblelist.append(string)
    return np.asarray(ensemblelist)

def get_beta_value(b):
    if b == 'A':
        return 1.90
    elif b == 'B':
        return 1.95
    elif b == 'D':
        return 2.10
    else:
        print('bet not known')

def get_mul_value(l):
    return float(l)/10**4

def ensemblevalues(ensemblelist):
    """Convert array of ensemblenames to list of value tuples
    """
    ix_values = []
    for i,e in enumerate(ensemblelist):
        b = get_beta_value(e[0])
        l = int(e.split('.'[-1]))
        mul = get_mul_value(e[1:3])
        ix_values.append((b,l,mul))
    return(np.asarray(ix_values))
