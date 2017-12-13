#!/hiskp2/werner/libraries/Python-2.7.12/python

import pandas as pd
# TODO: Calling is complicated improve that
import chiron

def main():
    pd.set_option('display.width',1000)
    pd.set_option('display.precision',10)
    delim = '\n'+'-'*80+'\n'
    resdir="/hiskp2/helmes/analysis/scattering/test/pi_k/I_32/results"
    artefact_files=[
        "/pik_disc_eff_M1A.pkl",
        "/pik_disc_eff_M1B.pkl",
        "/pik_disc_eff_M2A.pkl",
        "/pik_disc_eff_M2B.pkl" 
        ]
    chiral_data = chiron.get_dataframe_disk(directory=resdir,filenames=artefact_files)
    print(chiral_data.info())
    # We are interested in the mean values and standard deviations for each
    # method, lattice artefact and fitrange separately
    groups = ['method', 'Lattice Artefact','fit_end']
    observables = ['L_5','L_piK','mu_a32_phys','chi2 reduced']
    bmeans = chiron.bootstrap_means(chiral_data,groups,observables)
    print(chiron.bootstrap_means(chiral_data,groups,
                                 observables).loc[(slice(None),'None'),:])
    # Furthermore we want to take the average over a few keys first and then
    # look at the mean values and standard deviations again.
    groups=['method','Lattice Artefact']
    observables=['L_5','L_piK','mu_a32_phys']
    fitrange_means=chiron.bootstrap_means_key(chiral_data,groups,observables,
                                              loc=(slice(None),'None'))
    chiron.print_si_format(fitrange_means)
    print(chiral_data.sample(n=25))
    # Next we need to take some averages to estimate systematics
    # Define a naive weight based on the fitranges

    # Weighted average of None-lattice artefact over fitranges and Zp for A and
    # B gives estimate of ms-fixing influence
    weighted_chiral_data = chiron.average_systematics(chiral_data,('M1A','M1B'),
                                                                 ('M2A','M2B'))

    # Weighted average of None-lattice artefact over fitranges and ms-fixing for
    # Zp1 and Zp2 fives estimate of renormalisation scale influence
    #weighted_chiral_data = chiron.average_systematics(chiral_data,('M1A','M2A'),
    #                                                              ('M1B','M2B'))
    
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Keyboard Interrupt")

