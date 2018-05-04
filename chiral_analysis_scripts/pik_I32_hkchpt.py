#!/usr/bin/python
# system imports
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

# Christian's packages
sys.path.append('/hiskp4/helmes/projects/analysis-code/')
import analysis2 as ana

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
    space=ens.get_data("beta")
    latA = ens.get_data("namea")
    latB = ens.get_data("nameb")
    latD = ens.get_data("named")
    strangeA = ens.get_data("strangea")
    strangeB = ens.get_data("strangeb")
    strangeD = ens.get_data("stranged")
    strange_eta_A = ens.get_data("strange_alt_a")
    strange_eta_B = ens.get_data("strange_alt_b")
    strange_eta_D = ens.get_data("strange_alt_d")
   # keep seeds per zp method fixed
    zp_meth=ens.get_data("zp_meth")
    try:
        epik_meth = ens.get_data("epik_meth")
    except:
        epik_meth=""
    external_seeds=ens.get_data("external_seeds_%s"%(ms_fixing.lower()))
    continuum_seeds=ens.get_data("continuum_seeds_%s"%(ms_fixing.lower()))
    latphys_seeds=ens.get_data("latphys_seeds_%s"%(ms_fixing.lower()))
    amulA = ens.get_data("amu_l_a")
    amulB = ens.get_data("amu_l_b")
    amulD = ens.get_data("amu_l_d")

    #dictionary of strange quark masses
    amusA = ens.get_data("amu_s_a")
    amusB = ens.get_data("amu_s_b")
    amusD = ens.get_data("amu_s_d")
    # dictionaries for chiral analysis
    lat_dict = ana.make_dict(space,[latA,latB,latD])
    amu_l_dict = ana.make_dict(space,[amulA,amulB,amulD])
    mu_s_dict = ana.make_dict(space,[strangeA,strangeB,strangeD])
    mu_s_eta_dict = ana.make_dict(space,[strange_eta_A,strange_eta_B,strange_eta_D])
    amu_s_dict = ana.make_dict(space,[amusA,amusB,amusD])
    print(amu_s_dict)
    #quark = ens.get_data("quark")
    datadir = ens.get_data("datadir") 
    plotdir = ens.get_data("plotdir") 
    resdir = ens.get_data("resultdir") 
    nboot = ens.get_data("nboot")
    # Prepare external data
    ext_data = ana.ExtDat(external_seeds,space,zp_meth)
    cont_data = ana.ContDat(continuum_seeds,zp_meth=zp_meth)
    latphys_data = ana.LatPhysDat(latphys_seeds,space,zp_meth)
    fpi_raw = ana.read_extern("../plots2/data/fpi.dat",(1,2))
    print(fpi_raw)

    lattice_artefacts={'None'                  :0}
    #                   '(a M_pi)^2'            :1,
    #                   '(a M_K)^2'             :2,
    #                   'a^2(M_K^2 + 0.5M_pi^2)':3,
    #                   '(a mu_piK)^2'          :4}
    error_functions={0: ana.wrap_test.err_func_hkchpt}

################### Setup chiral analysis ######################################
    extrapol = ana.ChirAna("pi-K_I32_chipt_%s%d_%s"
                           %(ms_fixing.upper(),zp_meth,epik_meth),
                           correlated=False,gamma=False,match=True,debug=2)
    # have five x-values: mk,mpi,meta,fpi,r0/a

    ens_shape_chirana = (len(latA),len(latB),len(latD))
    print(ens_shape_chirana)
    # Have four inputs mpi, mk, fpi and a
    lyt_xvals = ana.generate_chirana_shape(space,ens_shape_chirana,1,5,1500)
    lyt_yvals = ana.generate_chirana_shape(space,ens_shape_chirana,1,1,1500) 
    extrapol.create_empty(lyt_xvals,lyt_yvals,lat_dict=lat_dict)
    print("\nSetup complete, begin chiral analysis")
    try:
        extrapol.load(resdir)
    except:
        print("Could not load chiral analysis!")
    #for i,a in enumerate(space):
    #    for j,e in enumerate(lat_dict[a]):
    #        extrapol.x_data[i][j,0,3]*=latphys_data.get(a,'a')
    # initialize dataframe for storage
    # the columns are physical value, L_piK, L_5, c, artefact id
    observables=['fit_start','fit_end','chi2 reduced','A_piK',
                 'Alat_2','mu_a32_phys']
    results_lattice_artefact = pd.DataFrame(columns=observables)
    print(results_lattice_artefact.info())
    # Include bootstrapped L_5 as a prior value taken from HPQCD
    start = [0.1,0.1]
    for la in lattice_artefacts:
        print(80*'#')
        print('lattice artefact is %s'%la)
        print(80*'#')
        ranges=[[0,1.35],[0,1.41],[0,2.5]]
        filename = (plotdir+'/mu_a32_nlo_disc_eff_'+str(lattice_artefacts[la])
                   +'_M%d%s%s.pdf')%(zp_meth,ms_fixing.upper(),epik_meth)
        chiral_plot = ana.LatticePlot(filename, join = False,debug=4)
        for cut in ranges:
            print(80*'#')
            print('fit range is %r'%cut)
            print(80*'#')
            # fit results
            _error_function=error_functions[lattice_artefacts[la]] 
            extrapol.fit(_error_function,start,plotdir=plotdir,prior=None,
                         xcut=cut,pik=True)
            extrapol.mu_a0_pik_phys(cont_data.get('mpi_0'),cont_data.get('mk'),
                                         cont_data.get('fpi'),cont_data.get('r0')/197.37,
                                         iso_32=True,hkchpt=True)
            # build resultsarray
            st='|S%d'%len(la)
            chisquared_reduced = np.full(1500,
                                              extrapol.fit_stats[0][1]/extrapol.fit_stats[0][0])
            a_pik = extrapol.fitres.data[0][:,0,0]
            a_2 = extrapol.fitres.data[0][:,1,0]
            mu_a32_phys = extrapol.phys_point_fitres.data[0][:,0]
            fitrange_start = np.full_like(a_pik,cut[0])
            fitrange_end = np.full_like(a_pik,cut[1])
            value_list=[fitrange_start,fitrange_end,
                        chisquared_reduced,a_2,a_pik,mu_a32_phys]
            fitresults={ key:values for key, values in zip(observables,value_list) }
            fitresults_dataframe=pd.DataFrame(fitresults)
            results_lattice_artefact = results_lattice_artefact.append(fitresults_dataframe)
            # summary of data
            header = ["#amu_l", "amu_s", "mpi", "dmpi", "mk", "dmk", "fpi", "dfpi",
                      "r0/a", "d(r0/a)", "m_eta","dm_eta","mu a0", "d(mu a0)"]
            ana.print_summary(extrapol,header,amu_l_dict,amu_s_dict) 
            #Plot
            plotargs = extrapol.fitres.data[0]
            lat_space=[0.0885,0.0815,0.0619]
            args = np.asarray([np.vstack((plotargs[:,0,0],
              plotargs[:,1,0],np.zeros_like(plotargs[:,1,0]),
              np.full((1500),lat_space[r]),cont_data.get('l4'),
              cont_data.get('b0'))).T for r in range(len(space))])
            # Plot the data the given function
            label=[r'$\mu_{\pi K}/f_{\pi}$',r'$\mu_{\pi K}\,a_{0}$',r' ',
                   r'$(\mu_{\pi K}/f_{\pi})_{phys}$']
            chiral_plot.plot_chiral_ext(extrapol,space,label, xlim=[0.7,1.7],
                                        ylim=[-0.25,0.], func=None, args=args,
                                        calc_x = None, ploterror=False,
                                        kk=False, x_phys=0.8128,plotlim=[0.7,1.7],
                                        argct=None)
            args = np.vstack((plotargs[:,0,0], plotargs[:,1,0],
              np.zeros_like(plotargs[:,0,0]),np.ones_like(plotargs[:,0,0]),cont_data.get('l4'),cont_data.get('b0'))).T
            chiral_plot.plot_cont(extrapol,ana.wrap_test.pik_I32_lo,
                                  [0.7,1.7],args,argct=None,calc_x=None,ploterror=False,
                                  label=r'LO-$\chi$-PT',xcut=cut)
            chiral_plot.save()
            label[1]=r'rel. dev. $\mu_{\pi K}\,a_{0}$'
            chiral_plot.plot_fit_proof(extrapol,space,ana.wrap_test.mu_pik_a_32_hkchpt_fit,
                                        xvalue_function=ana.calc_x_plot,
                                        data_label='fit',plotlim=[1.1,1.6],ylim=[-0.3,0.3],label=label,
                                        x_phys=None)
            chiral_plot.save()
        del chiral_plot
    #print(results_lattice_artefact.info())
    process_id = 'pik_hkchpt_M%d%s_%s'%(zp_meth, ms_fixing.upper(),epik_meth)
    hdf_filename = resdir+process_id+'.h5'
    hdfstorer = pd.HDFStore(hdf_filename)
    hdfstorer[process_id] = results_lattice_artefact 
    del hdfstorer
    #results_lattice_artefact.to_pickle(resdir+'/pik_disc_eff_M%d%s.pkl'%(zp_meth,
    #                                   ms_fixing.upper()))

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Keyboard Interrupt")
