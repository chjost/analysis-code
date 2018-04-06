#!/usr/bin/python

import pandas as pd
# TODO: Calling is complicated improve that
import chiron

def main():
    pd.set_option('display.width',1000)
    pd.set_option('display.precision',10)
    delim = '\n'+'-'*80+'\n'
    method_delim = '#'*80
    #resdir="/hiskp4/hiskp2/helmes/analysis/scattering/test/pi_k/I_32/results"
    resdir="/hiskp4/helmes/analysis/scattering/pi_k/I_32_final/results"
    epik_meth="E1"
################################################################################
#
#     Analysis for NLO ChPT
#
################################################################################
    print('\n'+method_delim)
    print("Analysing systematic effects for ChPT fit at NLO:")
    print(method_delim+'\n')
    artefact_files=[
        "/pik_disc_eff_M1A_%s.h5"%epik_meth,
        "/pik_disc_eff_M1B_%s.h5"%epik_meth,
        "/pik_disc_eff_M2A_%s.h5"%epik_meth,
        "/pik_disc_eff_M2B_%s.h5"%epik_meth 
        ]
    chiral_data = chiron.get_dataframe_disk(directory=resdir,filenames=artefact_files)
    #print(chiral_data.info())
    # We are interested in the mean values and standard deviations for each
    # method, lattice artefact and fitrange separately
    groups = ['method', 'Lattice Artefact','fit_end']
    observables = ['L_5','L_piK','mu_a32_phys','chi2 reduced']
    bmeans = chiron.bootstrap_means(chiral_data,groups,observables)
    chiron.print_si_format(chiron.bootstrap_means(chiral_data,groups,
                                 observables).loc[(slice(None),'None'),:])
    # Furthermore we want to take the average over a few keys first and then
    # look at the mean values and standard deviations again.
    groups=['method','Lattice Artefact']
    observables=['L_5','L_piK','mu_a32_phys']
    fitrange_means=chiron.bootstrap_means_key(chiral_data,groups,observables,
                                              loc=(slice(None),'None'))
    chiron.print_si_format(fitrange_means)
    # Next we need to take some averages to estimate systematics
    # Define a naive weight based on the fitranges
    method_mean = chiron.average_all_methods(chiral_data,('M1A','M2A',
                                                          'M1B','M2B')) 

    # Weighted average of None-lattice artefact over fitranges and Zp for A and
    # B gives estimate of ms-fixing influence
    systematic_ms = chiron.average_methods(chiral_data,('M1A','M1B'),
                                           ('M2A','M2B'), fixed=['Zp1','Zp2'])

    # Weighted average of None-lattice artefact over fitranges and ms-fixing for
    # Zp1 and Zp2 fives estimate of renormalisation scale influence
    systematic_zp = chiron.average_methods(chiral_data,('M1A','M2A'),
                                           ('M1B','M2B'), fixed=['A','B'])

    # build dataframe with final systematic results
    final_result=chiron.combine(method_mean,sys=[systematic_ms, systematic_zp],
                                names=['fix_ms','Z_P'])
    print(delim)
    print("Final numbers for systematic analysis:")
    print(final_result)
    print(delim)
################################################################################
#
#     Analysis for Gamma-Method
#
################################################################################
    print('\n'+method_delim)
    print("Analysing systematic effects for Gamma fit:")
    print(method_delim+'\n')
    artefact_files=[
        "/pik_gamma_M1A_%s.h5"%epik_meth,
        "/pik_gamma_M1B_%s.h5"%epik_meth,
        "/pik_gamma_M2A_%s.h5"%epik_meth,
        "/pik_gamma_M2B_%s.h5"%epik_meth 
        ]
    chiral_data = chiron.get_dataframe_disk(directory=resdir,filenames=artefact_files)
    #print(chiral_data.info())
    # We are interested in the mean values and standard deviations for each
    # method, lattice artefact and fitrange separately
    groups = ['method', 'fit_start']
    observables = ['L_5','L_piK','mu_a32_phys','chi2 reduced']
    bmeans = chiron.bootstrap_means(chiral_data,groups,observables)
    chiron.print_si_format(chiron.bootstrap_means(chiral_data,groups,
                                 observables).loc[(slice(None),),:])
    print(chiron.bootstrap_means(chiral_data,groups,
                                 observables))
    # Furthermore we want to take the average over a few keys first and then
    # look at the mean values and standard deviations again.
    groups=['method']
    observables=['L_5','L_piK','mu_a32_phys']
    fitrange_means=chiron.bootstrap_means_key(chiral_data,groups,observables)
    chiron.print_si_format(fitrange_means)
    # Next we need to take some averages to estimate systematics
    # Define a naive weight based on the fitranges
    method_mean = chiron.average_all_methods(chiral_data,('M1A','M2A',
                                                          'M1B','M2B'), 
                                           index=['fit_start'],value=[0.],drop=[])
    method_mean.info()

    # Weighted average of None-lattice artefact over fitranges and Zp for A and
    # B gives estimate of ms-fixing influence
    systematic_ms = chiron.average_methods(chiral_data,('M1A','M1B'),
                                           ('M2A','M2B'), fixed=['Zp1','Zp2'],
                                           index=['fit_start'],value=[0.],drop=[])

    # Weighted average of None-lattice artefact over fitranges and ms-fixing for
    # Zp1 and Zp2 fives estimate of renormalisation scale influence
    systematic_zp = chiron.average_methods(chiral_data,('M1A','M2A'),
                                           ('M1B','M2B'), fixed=['A','B'],
                                           index=['fit_start'],value=[0.],drop=[])

    # build dataframe with final systematic results
    final_result=chiron.combine(method_mean,sys=[systematic_ms, systematic_zp],
                                names=['fix_ms','Z_P'])
    print(delim)
    print("Final numbers for systematic analysis:")
    print(final_result)
    print(delim)
if __name__ == '__main__':                
    try:
        main()
    except KeyboardInterrupt:
        print("Keyboard Interrupt")

