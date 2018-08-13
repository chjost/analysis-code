#!/usr/bin/python
# Concatenate all fit result dataframes in one dataframe for all ensembles 
# ensemble for E1 and E3

import argparse
import numpy as np
import pandas as pd
import sys
# Christian's packages
sys.path.append('/hiskp4/helmes/projects/analysis-code/')
import analysis2 as ana
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

def get_mu_s_value(s):
    sstring = '0.0'+s.split('_')[-1]
    return float(sstring)

def ensemblevalues(ensemblelist):
    """Convert array of ensemblenames to list of value tuples
    """
    print(ensemblelist)
    ix_values = []
    for i,e in enumerate(ensemblelist):
        print(e)
        b = get_beta_value(e[0])
        l = int(e.split('.')[-1])
        mul = get_mul_value(e[1:3])
        ix_values.append((b,l,mul))
    return(np.asarray(ix_values))


def main():
    ens =["A30.32", "A40.24", "A40.32", "A60.24", "A80.24", "A100.24",
        "B35.32","B55.32","B85.24", 
        "D45.32", "D30.48"]
    mus_a_fld = ["amu_s_185","amu_s_225","amu_s_2464"]
    mus_b_fld = ["amu_s_16","amu_s_186","amu_s_21"]
    mus_d_fld = ["amu_s_13","amu_s_15","amu_s_18"]
    mus_d_fld_var = ["amu_s_115","amu_s_15","amu_s_18"]
#--------------- Define filenames
    path='/hiskp4/helmes/analysis/scattering/pi_k/I_32_publish'
    datadir = '%s/%s'%(path,'data')
    fit_pik_out = "fit_pik.h5"
    collect_out = 'fit_pik_collect.h5'
    # E1/2
    keyname = 'fit_corr_e2_corr_false'
    e='A40.24'
    s='amu_s_185'
    fname = "%s/%s/%s/%s"%(datadir,e,s,fit_pik_out)
    fitres = pd.read_hdf(fname,key=keyname)
    collect_colnames = np.append(fitres.columns.values,['beta','L','mu_l','mu_s'])
    collect = pd.DataFrame(columns=collect_colnames)
    space = ['A','B','D','Dvar']
    ensemble={'A':["A30.32", "A40.24", "A40.32", "A60.24", "A80.24", "A100.24"],
            'B':["B35.32","B55.32","B85.24"],
            'D':["D45.32"],'Dvar':["D30.48"]}
    strange={'A':["amu_s_185","amu_s_225","amu_s_2464"],
             'B':["amu_s_16","amu_s_186","amu_s_21"],
             'D':["amu_s_13","amu_s_15","amu_s_18"],
             'Dvar':["amu_s_115","amu_s_15","amu_s_18"]}

    for i,a in enumerate(space):
        for j,e in enumerate(ensemble[a]):
            for k,s in enumerate(strange[a]):
                fname = "%s/%s/%s/%s"%(datadir,e,s,fit_pik_out)
                ix = ensemblevalues([e,])
                ix_s = get_mu_s_value(s)
                tmp = pd.read_hdf(fname,key=keyname)
                tmp['beta'] = ix[0][0]
                tmp['L'] = ix[0][1]
                tmp['mu_l'] = ix[0][2]
                tmp['mu_s'] = ix_s
                collect=collect.append(tmp)
    print(collect.sample(n=10))
    outname = "%s/%s"%(datadir,collect_out)
    collect.to_hdf(outname,key=keyname)

if __name__ == '__main__':
    try:
        print("starting")
        main()
    except KeyboardInterrupt:
        pass

