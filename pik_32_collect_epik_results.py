#!/usr/bin/python
# Overall goal: Plot of the fitweights as a function of t_f for each t_i on one
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
    parser = argparse.ArgumentParser()

    parser.add_argument("--mus",help="bare strange quarkm mass",type=float,
                        required=True)
    parser.add_argument("--infile",help="infile for paths",type=str,
                        required=True)
    args=parser.parse_args()
    # Get presets from analysis input files
    ens = ana.LatticeEnsemble.parse(args.infile)
    # get data from input file
    prefix = ens.get_data("path")
    print prefix
    lat = ens.name()
    nboot = ens.get_data("nboot")
    datadir = ens.get_data("datadir")
    plotdir = ens.get_data("plotdir")
    gmax = ens.get_data("gmax")
    d2 = ens.get_data("d2")
    try:
        debug = ens.get_data("debug")
    except KeyError:
        debug = 0
    L = ens.L()
    T = ens.T()
    T2 = ens.T2()
    addT = np.ones((nboot,)) * T
    addT2 = np.ones((nboot,)) * T2
#--------------- Define filenames
    fit_k_out="fit_k"
    fit_pi_out = "fit_pi"
    fit_pik_out = "fit_pik"
    # place in a dataframe with columns:
    # beta, L, mu_l, mu_s, t_i, t_f, p-value, weight, sample, E_piK, E_pi-E_K, poll
    result = pd.DataFrame(columns=['beta', 'L', 'mu_l', 'mu_s', 't_i', 't_f',
                                   'p-value', 'weight', 'sample', 'E_piK',
                                   'poll'])
    # Load fitresults for one ensemble and all strange quark masses
    poll='E1'
    ana_fitres = ana.FitResult.read("%s/%s_%s_%s.npz" % (datadir,fit_pik_out, lat,poll))
    ana_fitres.calc_error()
    #ana_fitres.print_details()
    print(ana_fitres.data[0].shape)
    #print(ana_fitres.pval[0].shape)
    print(ana_fitres.weight[1])
    print(ana_fitres.fit_ranges.shape)
    #print(ana_fitres.data[0].shape)
    ix = ensemblevalues([lat])[0]
    for i,fr in enumerate(ana_fitres.fit_ranges[0]):
        # build a small dataframe with all necessary values
        ana_df = pd.DataFrame()
        ana_df['sample'] = np.arange(ana_fitres.data[0].shape[0])
        ana_df['E_piK'] = ana_fitres.data[0][:,1,0,i]
        ana_df['t_i'] = ana_fitres.fit_ranges[0,i,0]
        ana_df['t_f'] = ana_fitres.fit_ranges[0,i,1]
        ana_df['poll']=poll
        ana_df['beta']=ix[0]
        ana_df['L']=int(ix[1])
        ana_df['mu_l'] = ix[2]
        ana_df['mu_s'] = args.mus
        ana_df['p-value']=ana_fitres.pval[0][:,0,i]
        ana_df['weight']=ana_fitres.weight[1][0][0,i]
        result=result.append(ana_df)
    result.info()

    # Load fitresults for one ensemble and all strange quark masses
    poll='E3'
    ana_fitres = ana.FitResult.read("%s/%s_%s_%s.npz" % (datadir,fit_pik_out, lat,poll))
    ana_fitres.calc_error()
    #ana_fitres.print_details()
    print(ana_fitres.data[0].shape)
    print(ana_fitres.pval[0].shape)
    print(ana_fitres.weight[0][0])
    print(ana_fitres.fit_ranges.shape)
    for i,fr in enumerate(ana_fitres.fit_ranges[0]):
        # build a small dataframe with all necessary values
        ana_df = pd.DataFrame()
        ana_df['sample'] = np.arange(ana_fitres.data[0].shape[0])
        ana_df['E_piK'] = ana_fitres.data[0][:,0,i]
        ana_df['t_i'] = ana_fitres.fit_ranges[0,i,0]
        ana_df['t_f'] = ana_fitres.fit_ranges[0,i,1]
        ana_df['poll']=poll
        ana_df['beta']=ix[0]
        ana_df['L']=int(ix[1])
        ana_df['mu_l'] = ix[2]
        ana_df['mu_s'] = args.mus
        ana_df['p-value']=ana_fitres.pval[0][:,i]
        ana_df['weight']=ana_fitres.weight[0][0][i]
        result=result.append(ana_df)
    result.info()
    storer = pd.HDFStore(datadir+'/'+fit_pik_out+'.h5')
    storer.put('summary',result)
    del storer

if __name__ == '__main__':
    try:
        print("starting")
        main()
    except KeyboardInterrupt:
        pass
