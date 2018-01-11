#!/usr/bin/python2
import sys
from scipy import stats
from scipy import interpolate as ip
import numpy as np
from numpy.polynomial import polynomial as P
import pandas as pd
import math
import matplotlib
matplotlib.use('Agg') # has to be imported before the next lines
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages
import collections as coll
# Christian's packages
sys.path.append('/home/christopher/programming/analysis-code/')
import analysis2 as ana

def concat_data_cov(data, space, beta_vals, mu_l_vals, mu_s_vals, prior=None):
    # infer dimensions of covariance matrix
    # Bootstrapsamples
    nboot = data['sample'].nunique()
    nobs = 0
    for i,beta in enumerate(beta_vals):
       data_per_beta=data.where(data['beta'] == beta).dropna()
       nmu_s = data_per_beta['mu_s'].nunique()
       nmu_l = data_per_beta['ens_id'].nunique()
       nobs += nmu_l*nmu_s
    # Covariance matrix always be two dimensional
    preliminary_shape = (nobs,nboot)
    cov_data = data.as_matrix(['ratio']).reshape(preliminary_shape)
    if prior is not None:
        # requires prior to be 2d
        nobs += prior.shape[0]
    if prior is not None:
        cov = np.vstack((cov_data,prior))
    else:
        cov = cov_data
    return cov


#def concat_data_cov(data,prior=None,debug=1):
#    """ Change data layout for estimating covariance matrix
#
#    A covariance matrix is a 2d array of shape (nvar,nmeas)
#    data comes in as a list with layout of a ChirAna object
#
#    Parameters
#    ----------
#    lst: data of a chirana object
#    prior: possible array of priors, needs to be at least 2d with shape[1]=nboot
#    """
#    
#    # get layout values
#    # last shape entry is number of bootstrapsamples
#    nmeas = data['sample'].nunique()
#    nvar = 0
#    # at the moment only one y-observable is allowed 
#    for n in lst:
#        nvar += n.shape[0]*n.shape[1]
#    if prior is not None:
#        nvar += prior.shape[0]
#    # initialize data array
#    d = np.zeros((nvar,nmeas))
#    # fill array
#    offset = 0
#    for i,l in enumerate(beta_vals):
#        # get number of ensembles for each lattice spacing
#        ens = 
#        mu = l.shape[1]
#        for _ens in range(ens):
#            for _mu in range(mu):
#                index = offset + _ens *mu +_mu
#                # choose 0th entry of y-values
#                d[index] = l[_ens,_mu,0]
#        offset += ens*mu
#    if prior is not None:
#        d[offset:] = prior
#    return d
def concat_data_fit(data,space,beta_values,obs,prior=None,debug=1):
    """ Function to reorganise read in data

    The fit function loops over the bootstrapsamples, we would like to evaluate
    the fit function for every lattice spacing separately. Hence this function
    returns a list of the bootstrapsamples. Each bootstrapsample contains a
    namedtuple over the Lattice spacings. Each namedtuple entry is a numpy array
    of shape (nb_ens(beta), nb_dim). Optional a prior can be added as an
    additional named tuple.
    """

    # set up data structures
    # number of bootstrapsamples
    nboot = data['sample'].nunique()
    print("inferred R = %d"%nboot)
    #nboot = lst[0].shape[-1]
    # named tuple instance
    # modify space if prior is given
    # TODO: what happens with multiple priors?
    if prior is not None:
      space += 'p'
    beta = coll.namedtuple('beta',space)
    d = []
    if prior is not None:
        for b in range(nboot):
            sampleframe = data.where(data['sample']==b).dropna()
            # compile list of lattice spacings
            tmp = []
            for i in range(len(space)-1):
                subframe=sampleframe.where(sampleframe['beta']==beta_values[i]).dropna()
                ens = subframe.mu_l.unique()
                mu = subframe.mu_s.unique()
                tmp.append(subframe.as_matrix(obs))
            if len(prior) < nboot:
                _p = []
                for p in prior:
                    _p.append(p[b])
                tmp.append(_p)  
            else:
                tmp.append(prior[b])
            tmpnt = beta(*tmp)
            d.append(tmpnt)
    # Code doubling in favour of faster for loop not sure if needed anmore
    else:
        for b in range(nboot):
            sampleframe = data.where(data['sample']==b).dropna()
            # compile list of lattice spacings
            tmp = []
            for i in range(len(space)):
                subframe=sampleframe.where(sampleframe['beta']==beta_values[i]).dropna()
                ens = subframe.mu_l.unique()
                mu = subframe.mu_s.unique()
                tmp.append(subframe.as_matrix(obs))
            # TODO: Bad style think of something else
            tmpnt = beta(*tmp)
            d.append(tmpnt)
    return d

def main():
################################################################################
#                   set up objects                                             #
################################################################################
    # Get parameters from initfile
    if len(sys.argv) < 2:
        ens = ana.LatticeEnsemble.parse("A40.24.ini")
    else:
        ens = ana.LatticeEnsemble.parse(sys.argv[1])
    # second system argument is fixing for ms
    ms_fixing=sys.argv[2]
    # get data from input file
    lat = ens.name()
    latA = ens.get_data("namea")
    latB = ens.get_data("nameb")
    latD = ens.get_data("named")
    strangeA = ens.get_data("strangea")
    strangeB = ens.get_data("strangeb")
    strangeD = ens.get_data("stranged")
    strange_eta_A = ens.get_data("strange_alt_a")
    strange_eta_B = ens.get_data("strange_alt_b")
    strange_eta_D = ens.get_data("strange_alt_d")
    space=['A','B','D']
   # keep seeds per zp method fixed
    zp_meth=ens.get_data("zp_meth")
    external_seeds=ens.get_data("external_seeds_%s"%(ms_fixing.lower()))
    continuum_seeds=ens.get_data("continuum_seeds_%s"%(ms_fixing.lower()))
    lat_dict = {'A':latA,'B':latB,'D':latD}
    print("lat_dict['A'] reads:")
    print(lat_dict['A'])
    amulA = ens.get_data("amu_l_a")
    amulB = ens.get_data("amu_l_b")
    amulD = ens.get_data("amu_l_d")
    amu_l_dict = {'A': amulA,'B': amulB, 'D': amulD}

    #dictionary of strange quark masses
    mu_s_dict = {'A': strangeA,'B': strangeB, 'D': strangeD}
    mu_s_eta_dict = {'A': strange_eta_A,'B': strange_eta_B, 'D': strange_eta_D}
    amusA = ens.get_data("amu_s_a")
    amusB = ens.get_data("amu_s_b")
    amusD = ens.get_data("amu_s_d")
    amu_s_dict = {'A': amusA,'B': amusB, 'D': amusD}
    print(amu_s_dict)
    #quark = ens.get_data("quark")
    datadir = ens.get_data("datadir") 
    plotdir = ens.get_data("plotdir") 
    resdir = ens.get_data("resultdir") 
    nboot = ens.get_data("nboot")
    # Prepare external data
    ext_data = ana.ExtDat(external_seeds,space,zp_meth)
    cont_data = ana.ContDat(continuum_seeds,zp_meth=zp_meth)
    fpi_raw = ana.read_extern("../plots2/data/fpi.dat",(1,2))
    dummies=np.loadtxt("./dummy_data_fk_fpi.txt")
    # set up dummy data for experiments
    observables = ['beta','ens_id','mu_l','mu_s','sample','f_k','f_pi', 'M_pi','M_K','M_eta']
    results_fix_ms = pd.DataFrame(columns=observables)
    beta_vals = [1.90,1.95,2.1]
    for i,a in enumerate(space):
        for j,m in enumerate(amu_l_dict[a]):
            beta = np.full(nboot,beta_vals[i])
            mu_light = np.full(nboot,m)
            ens_id = np.full(nboot,lat_dict[a][j],dtype=object)
            value_list = [beta,ens_id,mu_light, amu_s_dict[a][0],np.arange(nboot),
                ana.draw_gauss_distributed(dummies[i+j,11],dummies[i+j,12],
                    (nboot,),origin=True),
                ana.draw_gauss_distributed(dummies[i+j,5],dummies[i+j,6],
                    (nboot,),origin=True),
                ana.draw_gauss_distributed(dummies[i+j,1],dummies[i+j,2],
                    (nboot,),origin=True),
                ana.draw_gauss_distributed(dummies[i+j,3],dummies[i+j,4],
                    (nboot,),origin=True),
                ana.draw_gauss_distributed(dummies[i+j,7],dummies[i+j,8],
                    (nboot,),origin=True)]
            tmp_frame=pd.DataFrame({key:values for key,
                                    values in zip(observables,value_list)})
            results_fix_ms = results_fix_ms.append(tmp_frame)
    print(results_fix_ms)
    #print(results_fix_ms.where(results_fix_ms['sample']==0).dropna())
# to reuse the fit functions already established organize the data again in
# named tuples
# We need two functions one organizing the named tuples 
    data_for_fit = concat_data_fit(results_fix_ms,space,beta_vals,
                                   ['f_k','f_pi', 'M_pi','M_K','M_eta'])
    print(data_for_fit)
# and one for the covariance matrix
    results_fix_ms['ratio'] = results_fix_ms['f_k']/results_fix_ms['f_pi']
    # Prior:
    prior=np.arange(6).reshape((3,2))
    data_for_cov = concat_data_cov(results_fix_ms, space, beta_vals, amu_l_dict,
            amu_s_dict, prior=prior)

if __name__=="__main__":
    try:
        main()
    except(KeyboardInterrupt):
        print("KeyboardInterrupt")
