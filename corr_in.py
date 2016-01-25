#!/hadron/knippsch/Enthought/Canopy_64bit/User/bin/python

import sys
import numpy as np
import os
import re
import analysis2 as ana

###############################################################################
############################### Start Main ####################################
###############################################################################

def miss_confs(path,rng):
  misslist = []
  for c in range(rng[0],rng[1]+1,rng[2]):
    tmp_path = path+"cnfg%d" %c 
    if os.path.exists(tmp_path) is False:
      misslist.append(c)
  return misslist

def main():

    # parse the input file
    if len(sys.argv) < 2:
        ens = ana.LatticeEnsemble.parse("charged.ini")
    else:
        ens = ana.LatticeEnsemble.parse(sys.argv[1])

    # get data from input file
    lat = ens.get_data("name")
    T = ens.get_data("T")
    rawdir = ens.get_data("rawdir") 
    datadir = ens.get_data("datadir") 
    # TODO: Place that in the ensemble class
    if len(sys.argv) < 2:
      Corrs = ana.inputnames('charged.ini',['C2+', 'C4+C', 'C4+D'])
    else:
      Corrs = ana.inputnames(sys.argv[1],['C2+', 'C4+C', 'C4+D'])

    print(rawdir)
    print(datadir)
    ## path to correlation functions
    #res_path = '/hiskp2/helmes/contractions/kaon_scattering/tests/s_3_rnd_vec/A40.24/strange_225/data/'
    ##res_path='/hiskp2/helmes/contractions/pion_scattering/A40.24/data/'
    ##Corrs = ana.inputnames('./charged.ini',['C2+'])
    ## path to analysis data
    #res ='/hiskp2/helmes/analysis/scattering/test/s_3_rnd_vec/k_charged/data/A40.24/amu_s_225/'
    ##res = '/hiskp2/helmes/analysis/scattering/pion_test/data/A40.24/' 
    inputlist = []
    #f = open('conf.txt','r')
    #x = f.read().split('\n')
    #del x[-1]
    #f.close()
    #inputlist = ['cnfg'+i+'/' for i in x]
    #print inputlist
    cfg_rng = [600,3105,8]
    missing = miss_confs(rawdir,cfg_rng)
    for i in range(cfg_rng[0],cfg_rng[1],cfg_rng[2]):
      if i in missing:
        continue
      inputlist.append('cnfg%d/' % i)
    print(len(inputlist))
    # Read in correlators
    print("Reading Correlation functions from %s..." % rawdir)
    print("C2")
    C2 = ana.read_confs(rawdir,Corrs[0],inputlist,T)
    print("C4")
    C4D = ana.read_confs(rawdir,Corrs[1],inputlist,T)
    C4C = ana.read_confs(rawdir,Corrs[2],inputlist,T)
    print("Read in done")
    # subtract crossed from direct diagram
    C4_tot = ana.confs_subtr(C4D,C4C)
    C4_tot = ana.confs_mult(C4_tot,2)
    print("Writing to: %s..." % datadir)
    ana.write_data_ascii(C2,datadir+'k_charged_p0.dat')
    ana.write_data_ascii(C4_tot,datadir+'kk_charged_A1_TP0_00.dat')
    #ana.write_data_ascii(C4D,datadir+'C4D.dat')
    #ana.write_data_ascii(C4C,datadir+'C4C.dat')
    print("Finished")


  
if __name__ == "__main__":
    main()

