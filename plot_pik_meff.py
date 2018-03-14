#!/usr/bin/python
# Plot all 4pt effective mass curves from E1-E3 in a reasonable resolution
import sys
sys.path.append("/hiskp4/helmes/projects/analysis-code")
import numpy as np
import matplotlib
matplotlib.use('Agg') # has to be imported before the next lines
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import fsolve

import analysis2 as ana
def mass_divided(data,epi,ek,T):
    _T = T
    mass = np.zeros_like(data[:,:-2])
    print("Exponential solve for symmetric correlator")
    print(mass.shape)
    for b, row in enumerate(data):
        for t in range(len(row)-2):
             mass[b, t] = fsolve(pik_div,0.5,
                                 args=(row[t],row[t+1],t,_T,epi[b],ek[b]))
    correlator_mass=ana.Correlators.create(mass,T=_T)
    return correlator_mass

def pik_pollution(t,T,epi,ek):
    pollution = np.exp(-epi*t) * np.exp(-ek*(T-t)) + np.exp(-ek*t) * np.exp(-epi*(T-t))
    return pollution

def pik_div(epik,row,row_shifted,t,T,epi,ek):
    num = np.exp(-epik*t)+np.exp(-epik*(T-t)) - pik_pollution(t,T,epi,ek)/pik_pollution(t+1,T,epi,ek) * (np.exp(-epik*(t+1))+np.exp(-epik*(T-(t+1))))
    den = np.exp(-epik*(t+1))+np.exp(-epik*(T-(t+1))) - pik_pollution(t+1,T,epi,ek)/pik_pollution(t+2,T,epi,ek) * (np.exp(-epik*(t+2))+np.exp(-epik*(T-(t+2))))
    return row/row_shifted - num/den

def plot_pik_meffs(ens_names,datafolder,match=False,strange_dict=None,
                  strange_eta_dict=None,syserror=True):
    """ pi-K summary of effective masses. In contrast to pipi summary one degree of freedom more
    (m_s)
    
    Parameters
    ----------
    ens_names : a list of the ensemble names for the summary
    datadir : string where to find the data
    match : bool, is data matched to some mu_s value or not?
    strange_dict : dictionary of strange quark masses used, if match is false
    """
    
    # Table contents: Ensemble amu_s(matched) M_K(matched) delta_E a_0 M_K*a_0
    print("Collect data for summary in unmatched case")
    for i,e in enumerate(ens_names):
        print("\nread data for %s" % e)
        datapath =  '/hiskp4/helmes/analysis/scattering/pi_k/I_32/data/'
        kaon_fit_file = 'fit_k_unit_%s.npz' %e
        pion_fit_file = 'fit_pi_unit_%s.npz' %e
        pik_data_file = 'pik_charged_A1_TP0_00'

        T = 2*int(e[-2:])
        print("T as given in input is: %d" %T)
        if e == "D30.48":
            strange_dict[e[0]] = ["amu_s_115","amu_s_15","amu_s_18"]
            strange_eta_dict[e[0]] = ["strange_1150", "strange_1500", "strange_1800"]
        for j,s in enumerate(strange_dict[e[0]]):
            datadir = datapath+e+'/'+s+'/'
            #get the three correlators for each mu_s
            kfit = ana.FitResult.read("%s/%s"%(datadir, kaon_fit_file))
            pifit = ana.FitResult.read("%s/%s"%(datadir, pion_fit_file))
            files = ["%s/%s.dat" % (datadir,pik_data_file)]
            pikcorr = ana.Correlators(files, matrix=False,conf_col=3)
            pikcorr.sym_and_boot(1500)
            # E_K and E_pi needed all over script
            e_k = kfit.singularize().data[0][:,1,0]
            e_pi = pifit.singularize().data[0][:,1,0]

            # calculate m_eff(pik) for E1-E3
            # E1
            # Before fitting shift the correlator
            # make matrix out of corr
            corr_shift = ana.Correlators.create(pikcorr.data)
            corr_shift.matrix=True
            corr_shift.shape = np.append(pikcorr.data.shape,1)
            corr_shift.data.reshape((corr_shift.shape))
            print(e_k[0],e_pi[0])
            corr_shift.shift(1,mass = e_k-e_pi, shift=1,d2=0)
            # Convert again to correlator for plotting ws denotes weighted and shifted
            e1_pikcorr = ana.Correlators.create(corr_shift.data[...,0],T=T)
            e1_pikcorr.shape = e1_pikcorr.data.shape
            # TODO: How to calculate mass
            e1_pikcorr.mass(usecosh=False,weight = e_k-e_pi,shift=1.,T=T)

            # E2
            e2_pikcorr = ana.Correlators.create(pikcorr.data,T=T)
            e2_pikcorr.subtract_pollution(pifit,kfit)
            e2_pikcorr.mass(usecosh=True)

            # E3
            # Before fitting shift the correlator
            # make matrix out of corr
            corr_shift_e3 = ana.Correlators.create(pikcorr.data)
            corr_shift_e3.matrix=True
            corr_shift_e3.shape = np.append(pikcorr.data.shape,1)
            corr_shift_e3.data.reshape((corr_shift_e3.shape))
            corr_shift_e3.shift(1, shift=1,d2=0)
            # Convert again to correlator for plotting ws denotes weighted and shifted
            corr_ws_e3 = ana.Correlators.create(corr_shift_e3.data[...,0],T=T)
            corr_ws_e3.shape = corr_ws_e3.data.shape
            corr_ws_e3.multiply_pollution(pifit,kfit)
            e3_pikcorr = mass_divided(corr_ws_e3.data,e_pi,e_k,T)
            
            # Plot the three methods together in one plot
            path = '/hiskp4/helmes/analysis/scattering/pi_k/quick_compare/m_effs'
            filename =path+'/compare_pik_meff_%s_%s.pdf'% (e,s) 
            pik_mass_plots = ana.LatticePlot(filename,join=True,debug=1)
            # Calculate plotlimits (ylimits are C(T/4-2)-0.05% and C(T/2)+0.05%)
            xlow, xhigh = T/8, T/2-3
            # new for every method

            ylow_e1 = e1_pikcorr.data[0,xlow,0]-0.05*e1_pikcorr.data[0,xlow,0]
            yhigh_e1 = e1_pikcorr.data[0,xhigh,0]+ 0.05*e1_pikcorr.data[0,xhigh,0]
            ylow_e2 = e2_pikcorr.data[0,xlow,0]-0.05*e2_pikcorr.data[0,xlow,0]
            yhigh_e2 = e2_pikcorr.data[0,xhigh,0]+ 0.05*e2_pikcorr.data[0,xhigh,0]
            ylow_e3 = e3_pikcorr.data[0,xlow,0]-0.05*e3_pikcorr.data[0,xlow,0]
            yhigh_e3 = e3_pikcorr.data[0,xhigh,0]+ 0.05*e3_pikcorr.data[0,xhigh,0]
            ylow = np.amin(np.asarray(ylow_e1,ylow_e2,ylow_e3))
            yhigh = np.amax(np.asarray(yhigh_e1,yhigh_e2,yhigh_e3))
            print("Plot limits are:")
            print("x: %d, %d"%(xlow, xhigh))
            print("y: %f, %f"%(ylow, yhigh))
            pik_mass_plots.set_env(ylog=False,grid=False,xlim=[xlow,xhigh],
                                   ylim=[ylow,yhigh])
            label = ['%s %s'%(e,s), 't', 'M_eff(t)','E1']
            pik_mass_plots.plot(e1_pikcorr,label)
            label = ['%s %s'%(e,s), 't', 'M_eff(t)','E2']
            pik_mass_plots.plot(e2_pikcorr,label)
            label = ['%s %s'%(e,s), 't', 'M_eff(t)','E3']
            pik_mass_plots.plot(e3_pikcorr,label)
            pik_mass_plots.save()
            del pik_mass_plots
            
def main():
  #ens =["A30.32", 
  #      "B85.24", "D45.32"]
  ens =["A30.32", "A40.24", "A40.32", "A60.24", "A80.24", "A100.24",
        "B35.32","B55.32","B85.24", 
        "D45.32", "D30.48"]
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
  data = '/hiskp4/helmes/analysis/scattering/pi_k/I_32/data/'
  plot_pik_meffs(ens,data,match=False,strange_dict = mass_fld,
                strange_eta_dict=mass_eta_fld,syserror=False)
# make this script importable, according to the Google Python Style Guide
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass

