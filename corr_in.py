#!/hadron/knippsch/Enthought/Canopy_64bit/User/bin/python


import numpy as np
import os
import re
import analysis as ana

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
  T = 48
  res_path='/hiskp2/helmes/contractions/kaon_scattering/B85.24/strange_186/data/'
  #res_path='/hiskp2/helmes/contractions/pion_scattering/A40.24/data/'
  Corrs = ana.inputnames('./charged.ini',['C2+', 'C4+C', 'C4+D'])
  #Corrs = ana.inputnames('./charged.ini',['C2+'])
  res = '/hiskp2/helmes/analysis/scattering/k_charged/data/B85.24/amu_s_186/' 
  #res = '/hiskp2/helmes/analysis/scattering/pion_test/data/A40.24/' 
  inputlist = []
  cfg_rng = [500,2900,8]
  missing = miss_confs(res_path,cfg_rng)
  for i in range(cfg_rng[0],cfg_rng[1],cfg_rng[2]):
    if i in missing:
      continue
    inputlist.append('cnfg%d/' % i)
  # Read in correlators
  print("Reading Correlation functions from %s..." % res_path)
  print("C2")
  C2 = ana.read_confs(res_path,Corrs[0],inputlist,T)
  print("C4")
  C4D = ana.read_confs(res_path,Corrs[1],inputlist,T)
  C4C = ana.read_confs(res_path,Corrs[2],inputlist,T)
  print("Read in done")
  #subtract crossed from direct diagram
  C4_tot = ana.confs_subtr(C4D,C4C)
  C4_tot = ana.confs_mult(C4_tot,2)
  print("Writing to: %s..." % res)
  ana.write_data_ascii(C2,res+'k_charged_p0.dat')
  #ana.write_data_ascii(C4C,res+'C4C_p0.0714.dat')
  #ana.write_data_ascii(C4D,res+'C4D_p0.0714.dat')
  ana.write_data_ascii(C4_tot,res+'kk_charged_A1_TP0_00.dat')
  print("Finished")


  
if __name__ == "__main__":
    main()

