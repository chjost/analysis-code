#!/usr/bin/python
# Script to plot the correlators and effective masses for pi_K

import sys
import numpy as np
import analysis2 as ana
def plot_pik_meff_e3(ensemble_list, datapath,strange_dict=None,
                     plotfile='./Meff_pik_e3.pdf'):
    nboot=1500
    # All plots should be in one file
    plotter = ana.LatticePlot(plotfile)
    # General filenames for read in:
    for i,e in enumerate(ensemble_list):
        pion_fitresult_name = 'fit_pi_%s.npz' %e 
        kaon_fitresult_name = 'fit_k_%s.npz' %e
        pik_corr_name =  'corr_pik_%s.npz'%e
        if e == "D30.48":
            strange_dict[e[0]] = ["amu_s_115","amu_s_15","amu_s_18"]
    # read in pion and kaon fitresults
        # Careful! not valid if A40.20 is included
        T=2*int(e[-2:])
        print("total Time extent is %d" %T) 
        mpi = ana.FitResult.read("%s/%s/pi/%s" % (datapath, e, pion_fitresult_name))
        mpi.calc_error()
        mpi.print_data(1)
        for j,s in enumerate(strange_dict[e[0]]):
            mk = ana.FitResult.read("%s/%s/%s/%s" % (datapath, e, s, kaon_fitresult_name))
            mk.calc_error()
            mk.print_data(1)

            # Read Pik
            files='%s/%s/%s/%s' %(datapath,e,s,pik_corr_name)
            pik_corr = ana.Correlators.read(files)
            # calculate C_pik with E3
            pik_corr_e3 = ana.Correlators.create(pik_corr.data,T=T)
            pik_corr_e3.divide_out_pollution(mpi,mk)
            e3_corr_shift = ana.Correlators.create(pik_corr_e3.data)
            e3_corr_shift.matrix=True
            e3_corr_shift.shape = np.append(pik_corr_e3.data.shape,1)
            e3_corr_shift.data.reshape((e3_corr_shift.shape))
            e3_corr_shift.shift(1, shift=1,d2=0)
            # Convert again to correlator for plotting ws denotes weighted and shifted
            e3_corr_ws = ana.Correlators.create(e3_corr_shift.data[...,0],T=T)
            e3_corr_ws.shape = e3_corr_ws.data.shape
            e3_corr_ws.multiply_pollution(mpi,mk)
            add = [mpi.data[0][:,1,0],mk.data[0][:,1,0]]
            e3_corr_ws.mass(function=5,add=add)
            # plot effective mass
            label=[r'$C_{\pi K} M_{eff}$ on %s %s' %(e,s), r'$t$', r'$M_{eff}(t)$',
                     r'E3']
            m = e3_corr_ws.data[0,int(T/3),0] 
            ylim = [m-0.05*m,m+0.05*m]
            xlim = [T/5,T/2-1]
            plotter.set_env(ylog=False,grid=False, title=True, ylim=ylim,
                    xlim=xlim)
            plotter.plot(e3_corr_ws,label)
    plotter.save()    
    del plotter 

def plot_pik_meff_e1(ensemble_list, datapath,strange_dict=None,
                     plotfile='./Meff_pik_e1.pdf'):
    nboot=1500
    # All plots should be in one file
    plotter = ana.LatticePlot(plotfile)
    # General filenames for read in:
    for i,e in enumerate(ensemble_list):
        pion_fitresult_name = 'fit_pi_%s.npz' %e 
        kaon_fitresult_name = 'fit_k_%s.npz' %e
        pik_corr_name =  'corr_pik_%s.npz'%e
        if e == "D30.48":
            strange_dict[e[0]] = ["amu_s_115","amu_s_15","amu_s_18"]
    # read in pion and kaon fitresults
        # Careful! not valid if A40.20 is included
        T=2*int(e[-2:])
        print("total Time extent is %d" %T) 
        mpi = ana.FitResult.read("%s/%s/pi/%s" % (datapath, e, pion_fitresult_name))
        mpi.calc_error()
        mpi.print_data(1)
        for j,s in enumerate(strange_dict[e[0]]):
            mk = ana.FitResult.read("%s/%s/%s/%s" % (datapath, e, s, kaon_fitresult_name))
            mk.calc_error()
            mk.print_data(1)

            # combine them in an energy difference
            diff_pi_k = mk.add_mass(mpi,neg=True)
            print(diff_pi_k.derived)
            diff_pi_k.singularize()
            diff_pi_k.print_details()
            diff_pi_k.print_data()
            # Read Pik
            files='%s/%s/%s/%s' %(datapath,e,s,pik_corr_name)
            pik_corr = ana.Correlators.read(files)
            # calculate C_pik with e1
            pik_corr_e1 = ana.Correlators.create(pik_corr.data,T=T)
            e1_corr_shift = ana.Correlators.create(pik_corr_e1.data)
            e1_corr_shift.matrix=True
            e1_corr_shift.shape = np.append(pik_corr_e1.data.shape,1)
            e1_corr_shift.data.reshape((e1_corr_shift.shape))
            e1_corr_shift.shift(1,mass=diff_pi_k.singularize().data[0][:,0,0], shift=1,d2=0)
            # Convert again to correlator for plotting ws denotes weighted and shifted
            e1_corr_ws = ana.Correlators.create(e1_corr_shift.data[...,0],T=T)
            e1_corr_ws.shape = e1_corr_ws.data.shape
            add = [mpi.data[0][:,1,0],mk.data[0][:,1,0]]
            e1_corr_ws.mass(function=4,add=add)
            # plot effective mass
            label=[r'$C_{\pi K} M_{eff}$ on %s %s' %(e,s), r'$t$', r'$M_{eff}(t)$',
                     r'e1']
            m = e1_corr_ws.data[0,int(T/3),0] 
            ylim = [m-0.05*m,m+0.05*m]
            xlim = [T/5,T/2-1]
            plotter.set_env(ylog=False,grid=False, title=True, ylim=ylim,
                    xlim=xlim)
            plotter.plot(e1_corr_ws,label)
    plotter.save()
    del plotter

def main():
    # Loop over ensembles and strange quark masses
    ens =["A30.32", "A40.24", "A40.32", "A60.24", "A80.24", "A100.24",
         "B35.32","B55.32","B85.24", 
         "D45.32","D30.48"]
    mus_a_fld = ["amu_s_185","amu_s_225","amu_s_2464"]
    mus_b_fld = ["amu_s_16","amu_s_186","amu_s_21"]
    mus_d_fld = ["amu_s_13","amu_s_15","amu_s_18"]
    mus_d_fld_var = ["amu_s_115","amu_s_15","amu_s_18"]
    mus_eta_a_fld =     ["strange_1850","strange_2250","strange_2464"]
    mus_eta_b_fld =     ["strange_1600","strange_1860","strange_2100"]
    mus_eta_d_fld =     ["strange_1300","strange_1500","strange_1800"]
    mus_eta_d_fld_var = ["strange_1150","strange_1500","strange_1800"]
    mass_fld = {"A":mus_a_fld,"B":mus_b_fld,"D":mus_d_fld}
    mass_eta_fld = {"A":mus_eta_a_fld,"B":mus_eta_b_fld,"D":mus_eta_d_fld}
    datapath = '/hiskp4/helmes/analysis/scattering/pi_k/I_32_blocked/data/'
    plot = '/hiskp4/helmes/analysis/scattering/pi_k/I_32_blocked/plots/'
    plot_pik_meff_e3(ens,datapath,strange_dict=mass_fld,plotfile=plot+"Meff_pik_e3.pdf")
    plot_pik_meff_e1(ens,datapath,strange_dict=mass_fld,plotfile=plot+"Meff_pik_e1.pdf")

if __name__ == '__main__':
    try:
        print("starting")
        main()
    except KeyboardInterrupt:
        pass


