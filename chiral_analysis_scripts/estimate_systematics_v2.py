#!/usr/bin/python

import itertools as it
import numpy as np
import pandas as pd
# TODO: Calling is complicated improve that
import analysis2 as ana
import chiron

def main():
    pd.set_option('display.width',1000)
    pd.set_option('display.precision',10)
    delim = '\n'+'-'*80+'\n'
    method_delim = '#'*80
    #resdir="/hiskp4/hiskp2/helmes/analysis/scattering/test/pi_k/I_32/results"
    resdir="/hiskp4/helmes/analysis/scattering/pi_k/I_32_final/results"
    
    epik_meth = ["E1","E3"]
    ms_meth = ['A','B']
    zp_meth = ['1','2']
    # Medians over fitrange into one dataframe
    raw_data = pd.DataFrame(columns = ['L_5','L_piK','mu_a32_phys', 'E_piK','Z_P','m_s','ChPT'])
    # build dataname
    for tp in it.product(epik_meth,ms_meth,zp_meth):
        name_list = ["/pik_gamma_M%s%s_%s.h5" %(tp[2],tp[1].upper(),tp[0])]
        # load file
        chiral_data = chiron.get_dataframe_disk(directory=resdir,
                                                filenames = name_list)
        chiral_data['E_piK'] = tp[0]
        chiral_data['ChPT'] = 'gamma'
        chiral_data['m_s'] = tp[1]
        chiral_data['Z_P'] = tp[2]
        chiral_data.info()
        raw_data = raw_data.append(chiral_data)

        name_list = ["/pik_disc_eff_M%s%s_%s.h5" %(tp[2],tp[1].upper(),tp[0])]
        # load file
        chiral_data = chiron.get_dataframe_disk(directory=resdir,
                                                filenames = name_list)
        chiral_data['E_piK'] = tp[0]
        chiral_data['ChPT'] = 'nlo'
        chiral_data['m_s'] = tp[1]
        chiral_data['Z_P'] = tp[2]
        chiral_data.info()
        raw_data = raw_data.append(chiral_data.where(chiral_data['Lattice Artefact']=='None'))
    
    raw_data['fit_interval'] = raw_data['fit_end']-raw_data['fit_start']
    raw_data = raw_data.drop(['method','Lattice Artefact',
        'c','fit_end','fit_start','p_val','Mpi_a12','Mpi_a32','tau'],axis=1)
    raw_data.info()
    # Get physical fpi, mpi and mk
    seeds = (3891,8612,3646,6548,6477,2357) 
    cont_dat = ana.ContDat(seeds,zp_meth="phys")
    delta_K = ana.draw_gauss_distributed(0.04,0.022,(1500,),origin=True)
    p_star = np.full_like(delta_K,11.8)
    data_in = {'mpi_0':cont_dat.get('mpi_0'),'mk':cont_dat.get('mk'),
            'fpi':cont_dat.get('fpi'),'delta_K':delta_K,'p_star':p_star}
    physical_input = pd.DataFrame(data_in)

    # put it into raw_data
    raw_data = raw_data.join(physical_input)
    raw_data.info()
    sources = ['E_piK','m_s','Z_P','ChPT']
    observables = ['L_piK','mu_a32_phys']
    final_result = chiron.get_systematics(raw_data,sources,observables)
    print(final_result)
    x = raw_data.as_matrix(columns = ['fpi','mpi_0','mk','L_piK','L_5','delta_K','p_star'])
    # Todo can this be made more efficient?
    raw_data['a_pos'] = pd.Series(data = ana.a_pik_pos(x[:,0],x[:,1],x[:,2],
                                                    x[:,0],x[:,3]),
                               index = raw_data.index)
    raw_data['a_neg'] = pd.Series(data = ana.a_pik_neg(x[:,0],x[:,1],x[:,2],
                                                    x[:,0],x[:,4]),
                               index = raw_data.index)
    # Calculate Mpi_a12 and Mpi_a32
    raw_data['Mpi_a32'] = raw_data['mpi_0']*(raw_data['a_pos']-raw_data['a_neg'])
    raw_data['Mpi_a12'] = raw_data['mpi_0']*(raw_data['a_pos']+2*raw_data['a_neg'])
    raw_data['Mpi_aneg'] = raw_data['mpi_0']*raw_data['a_neg']
    mpi_a = raw_data.as_matrix(columns = ['Mpi_a12','Mpi_a32'])
    raw_data['tau'] = pd.Series(data = ana.pi_k_tau_pandas(mpi_a[:,0],mpi_a[:,1],x[:,5],
                                                           x[:,1],x[:,2],x[:,6]),
                                                           index = raw_data.index)
    sources = ['E_piK','m_s','Z_P','ChPT']
    observables = ['L_piK','mu_a32_phys', 'Mpi_a32','Mpi_a12','tau','Mpi_aneg']
    final_result = chiron.get_systematics(raw_data,sources,observables)
    print(final_result)

if __name__ == '__main__':                
    try:
        main()
    except KeyboardInterrupt:
        print("Keyboard Interrupt")

