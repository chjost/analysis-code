#!/hadron/knippsch/Enthought/Canopy_64bit/User/bin/python
# A sript to get all correlationfunctions as ascii to the right position without
# loosing correlation in MC-time
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
      print(tmp_path)
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
    rawstrange = ens.get_data("raw_a")
    strange = ens.get_data("strangea")
    # TODO: Place that in the ensemble class
    if len(sys.argv) < 2:
      Corrs = ana.inputnames('charged.ini',['C20', 'C40C', 'C40D'])
    else:
      Corrs = ana.inputnames(sys.argv[1],['C20', 'C40C', 'C40D'])
    # os.path.join treats preceding slashes as new paths
    print(Corrs)
    corrpaths = [os.path.join(rawdir,mu_s,'data/') for mu_s in rawstrange]
    datapaths = [os.path.join(rawdir,mu_s,'data/') for mu_s in strange]
    # get all available correlators in three lists
    configs_coll = [os.listdir(cp) for cp in corrpaths]
    conf_feed = [i +'/' for i in set(configs_coll[0]).intersection(set(configs_coll[1]),set(configs_coll[2]))]
    print(conf_feed)
    for s in zip(corrpaths,datapaths):
      # copy common subset of configurations to appropriate target directory:
      # Read in correlators
      print("Reading Correlation functions from %s..." % s[0])
      print("C2")
      C2 = ana.read_confs(s[0],Corrs[0],conf_feed,T)
      print("C4")
      C4D = ana.read_confs(s[0],Corrs[1],conf_feed,T)
      C4C = ana.read_confs(s[0],Corrs[2],conf_feed,T)
      print("Read in done")
      # subtract crossed from direct diagram
      C4_tot = ana.confs_subtr(C4D,C4C)
      C4_tot = ana.confs_mult(C4_tot,2)
      print("Writing to: %s..." % s[1])
      #ana.write_data_ascii(C2,s[1]+'pi_charged_p0.dat')
      ana.write_data_ascii(C2,s[1]+'k_charged_p0.dat',conf_feed)
      ana.write_data_ascii(C4_tot,s[1]+'kk_charged_A1_TP0_00.dat',conf_feed)
      ana.write_data_ascii(C4D,s[1]+'C4D.dat')
      ana.write_data_ascii(C4C,s[1]+'C4C.dat')
    
print("Finished")
      
      

  

if __name__ == "__main__":
    main()
