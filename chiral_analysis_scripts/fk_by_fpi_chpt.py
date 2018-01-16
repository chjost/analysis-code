#!/usr/bin/python2
import sys
from scipy import stats
from scipy import interpolate as ip
import numpy as np
from numpy.polynomial import polynomial as P
import pandas as pd
import math
import matplotlib
#matplotlib.use('Agg') # has to be imported before the next lines
matplotlib.use('pgf') # has to be imported before the next lines
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages
import collections as coll
# Christian's packages
sys.path.append('/home/christopher/programming/analysis-code/')
import analysis2 as ana
import chiron as chi

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
    _space = space + ['p']
    beta = coll.namedtuple('beta',_space)
    d = []
    for b in range(nboot):
        sampleframe = data.where(data['sample']==b).dropna()
        # compile list of lattice spacings
        tmp = []
        for i in range(len(space)):
            subframe=sampleframe.where(sampleframe['beta']==beta_values[i]).dropna()
            ens = subframe.mu_l.unique()
            mu = subframe.mu_s.unique()
            subframe_to_add = subframe.as_matrix(obs) 
            if len(obs) == 1:
                subframe_to_add = subframe_to_add.flatten()
            tmp.append(subframe_to_add)
        if prior is not None:
            if len(prior) < nboot:
                _p = []
                for p in prior:
                    _p.append(p[b])
                tmp.append(_p)  
            else:
                tmp.append(prior[b])
        else:
            tmp.append(None)
        tmpnt = beta(*tmp)
        d.append(tmpnt)
    return d

def F(dataframe,col):
    mpi=dataframe[col[0]]
    mk=dataframe[col[1]]
    meta=dataframe[col[2]]
    fk=dataframe[col[3]]
    fpi=dataframe[col[4]]
    mu=dataframe[col[4]]
    f=dataframe[col[4]]
    F=ana.chipt_decayconstants.l5(mpi,mk,meta,fk,fpi,f,mu)
    return F

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
    ext_data = ana.ExtDat(external_seeds,space,zp_meth,nboot=nboot)
    cont_data = ana.ContDat(continuum_seeds,zp_meth=zp_meth)
    fpi_raw = ana.read_extern("../plots2/data/fpi.dat",(1,2))
    dummies=np.loadtxt("./dummy_data_fk_fpi.txt")
    # set up dummy data for experiments
    observables = ['beta','r0/a','ens_id','mu_l','mu_s','sample','f_k','f_pi', 'M_pi','M_K','M_eta']
    results_fix_ms = pd.DataFrame(columns=observables)
    beta_vals = [1.90,1.95,2.1]
    for i,a in enumerate(space):
        for j,m in enumerate(amu_l_dict[a]):
            beta = np.full(nboot,beta_vals[i])
            r_0 = ext_data.get(a,'r0')
            mu_light = np.full(nboot,m)
            ens_id = np.full(nboot,lat_dict[a][j],dtype=object)
            value_list = [beta, r_0, ens_id, mu_light, amu_s_dict[a][0],
                          np.arange(nboot),
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
    #xdata_for_fit = concat_data_fit(results_fix_ms,space,beta_vals,
    #                               ['f_k','f_pi', 'M_pi','M_K','M_eta'])
# and one for the covariance matrix
    results_fix_ms['ratio'] = results_fix_ms['f_k']/results_fix_ms['f_pi']
    results_fix_ms['F'] = F(results_fix_ms,['M_pi','M_K','M_eta','f_k','f_pi'])
    #ydata_for_fit = concat_data_fit(results_fix_ms,space,beta_vals,
    #                               ['ratio'])
    data_means = chi.syseffos.bootstrap_means(results_fix_ms,
                                                  ['ens_id'],['beta','r0/a','ens_id','mu_l','sample','f_k','f_pi',
                                                      'M_pi','M_K','M_eta','ratio','F'])
    #print(xdata_for_fit)
    #print(ydata_for_fit)
    # Prior:
    prior=np.arange(6).reshape((3,2))
    data_for_cov = concat_data_cov(results_fix_ms, space, beta_vals, amu_l_dict,
            amu_s_dict, prior=None)
    fit_instance=ana.ChiralFit("fit",ana.fk_fpi_ratio_errfunc)
    #_cov = np.cov(data_for_cov)
    #print(_cov)
    #_cov = np.diag(np.diagonal(_cov))
    start=[0.1]
    #fitres = fit_instance.chiral_fit(xdata_for_fit,ydata_for_fit,start,parlim=None,
    #                              correlated=False,cov=_cov,
    #                              debug=4)
    #fitres.set_ranges(np.array([[[0,len(xdata_for_fit)]]]),[[1,]])
    # build fit_stats array
    #_chi2 = fitres.chi2[0][0,0]
    #_pval = fitres.pval[0][0,0]
    #_dof = _cov.shape[0]-len(start)
    #fit_stats = np.atleast_2d(np.asarray((_dof,_chi2,_pval)))
    #print(fit_stats)

# We need a plot of the data points as a function of the light quark mass ->
# Same problem as for pi_K first plot data
    
    ydata_for_plot = chi.syseffos.bootstrap_means(results_fix_ms,
                                                  ['ens_id'],['beta','ratio'])
    results_fix_ms['(r0M_pi)^2'] = (results_fix_ms['r0/a']*results_fix_ms['M_pi'])**2
    xdata_for_plot = chi.syseffos.bootstrap_means(results_fix_ms,
                                                ['ens_id'],['beta','(r0M_pi)^2'])
    print(ydata_for_plot)
    print(xdata_for_plot)
    #with PdfPages('./dummy_ratio.pdf') as pdf:
    #    # set layout
    #    plt.ylabel(r'$f_K/f_{\pi}$')
    #    plt.xlabel(r'$(r_0M_{\pi})^2$')
    #    fmts=['^r','vb','og']
    #    for i,b in enumerate(beta_vals):
    #        x = xdata_for_plot.where(xdata_for_plot['beta','own_mean'] ==
    #                b)['(r0M_pi)^2','own_mean'].dropna()
    #        xerr = xdata_for_plot.where(xdata_for_plot['beta','own_mean'] ==
    #                b)['(r0M_pi)^2','own_std'].dropna()
    #        y = ydata_for_plot.where(ydata_for_plot['beta','own_mean'] ==
    #               b)['ratio','own_mean'].dropna()
    #        yerr = ydata_for_plot.where(ydata_for_plot['beta','own_mean'] ==
    #                b)['ratio','own_std'].dropna()
    #        
    #        plt.errorbar(x,y,yerr,xerr=xerr,fmt=fmts[i],label=r'$\beta=%.2f$'%b)
    #    plt.legend()
    #    pdf.savefig()
    plt.ylabel(r'$f_K/f_{\pi}$')
    plt.xlabel(r'$(r_0M_{\pi})^2$')
    fmts=['^r','vb','og']
    for i,b in enumerate(beta_vals):
        x = xdata_for_plot.where(xdata_for_plot['beta','own_mean'] ==
                b)['(r0M_pi)^2','own_mean'].dropna()
        xerr = xdata_for_plot.where(xdata_for_plot['beta','own_mean'] ==
                b)['(r0M_pi)^2','own_std'].dropna()
        y = ydata_for_plot.where(ydata_for_plot['beta','own_mean'] ==
               b)['ratio','own_mean'].dropna()
        yerr = ydata_for_plot.where(ydata_for_plot['beta','own_mean'] ==
                b)['ratio','own_std'].dropna()
        
        plt.errorbar(x,y,yerr,xerr=xerr,fmt=fmts[i],label=r'$\beta=%.2f$'%b)
    plt.legend()
    plt.savefig('dummy_ratio.pgf')

    ydata_for_plot = chi.syseffos.bootstrap_means(results_fix_ms,
                                                  ['ens_id'],['beta','F'])
    plt.clf()
    print(ydata_for_plot)
    # set layout
    plt.ylabel(r'$\mathcal{F}$')
    plt.xlabel(r'$(r_0M_{\pi})^2$')
    fmts=['^r','vb','og']
    for i,b in enumerate(beta_vals):
        x = xdata_for_plot.where(xdata_for_plot['beta','own_mean'] ==
                b)['(r0M_pi)^2','own_mean'].dropna()
        xerr = xdata_for_plot.where(xdata_for_plot['beta','own_mean'] ==
                b)['(r0M_pi)^2','own_std'].dropna()
        y = ydata_for_plot.where(ydata_for_plot['beta','own_mean'] ==
               b)['F','own_mean'].dropna()
        yerr = ydata_for_plot.where(ydata_for_plot['beta','own_mean'] ==
                b)['F','own_std'].dropna()
        
        plt.errorbar(x,y,yerr,xerr=xerr,fmt=fmts[i],label=r'$\beta=%.2f$'%b)
    plt.legend()
    plt.savefig('dummy_f.pgf')
    #with PdfPages('./dummy_f.pdf') as pdf:
    #    # set layout
    #    plt.ylabel(r'$\mathcal{F}$')
    #    plt.xlabel(r'$(r_0M_{\pi})^2$')
    #    fmts=['^r','vb','og']
    #    for i,b in enumerate(beta_vals):
    #        x = xdata_for_plot.where(xdata_for_plot['beta','own_mean'] ==
    #                b)['(r0M_pi)^2','own_mean'].dropna()
    #        xerr = xdata_for_plot.where(xdata_for_plot['beta','own_mean'] ==
    #                b)['(r0M_pi)^2','own_std'].dropna()
    #        y = ydata_for_plot.where(ydata_for_plot['beta','own_mean'] ==
    #               b)['F','own_mean'].dropna()
    #        yerr = ydata_for_plot.where(ydata_for_plot['beta','own_mean'] ==
    #                b)['F','own_std'].dropna()
    #        
    #        plt.errorbar(x,y,yerr,xerr=xerr,fmt=fmts[i],label=r'$\beta=%.2f$'%b)
    #    plt.legend()
    #    pdf.savefig()

if __name__=="__main__":
    try:
        main()
    except(KeyboardInterrupt):
        print("KeyboardInterrupt")
