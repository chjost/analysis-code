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
    chiral_data = chiron.SysEffos(directory=resdir,filenames=artefact_files,debug=2)
    # We are interested in the mean values and standard deviations for each
    # method, lattice artefact and fitrange separately
    groups=['method', 'Lattice Artefact','fit_end']
    observables=['L_5','L_piK','c','mu_a32_phys']
    chiral_data.bootstrap_means(groups,observables)
    print(chiral_data.mean_frame)
    # Furthermore we want to take the average over a few keys first and then
    # look at the mean values and standard deviations again.
    groups=['method','Lattice Artefact']
    chiral_data.bootstrap_means_key(groups,observables,mean_key='fit_end')
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Keyboard Interrupt")

