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
      print(tmp_path)
      misslist.append(c)
  return misslist

def main():
    
    read_c4=True
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
      Corrs = ana.inputnames(sys.argv[1],['c0','c1','c2','c3'])
      #Corrs = ana.inputnames(sys.argv[1],['c0','c1','c2','c3','c4','c5','c6','c7'])
      #Corrs = ana.inputnames(sys.argv[1],['c5'])

    print(rawdir)
    print(datadir)
    inputlist = []
    cfg_rng = [714,2330,4]
    #omit = [20, 164, 416, 540, 568, 596, 668, 1000]
    omit=[]
    print(omit)
    missing = miss_confs(rawdir,cfg_rng)
    missing.extend(omit)
    print (missing)
    for i in range(cfg_rng[0],cfg_rng[1]+1,cfg_rng[2]):
      if i in missing:
        print("omit cfg %d" % i)
        continue
      inputlist.append('cnfg%d/' % i)
    print inputlist
    print(len(inputlist))
    # Read in correlators
    print("Reading Correlation functions from %s..." % rawdir)
    print("C2")
    print(Corrs)
    C2_k = ana.read_confs(rawdir,Corrs[0],inputlist,T)
    C2_pi = ana.read_confs(rawdir,Corrs[1],inputlist,T)
    # Multiplication only needed for neutral function
    #C2_k = ana.confs_mult(C2_k,-1)
    #C2_pi = ana.confs_mult(C2_pi,-1)
    ana.write_data_ascii(C2_k,datadir+'k_unit.dat',conf=inputlist)
    ana.write_data_ascii(C2_pi,datadir+'pi_unit.dat',conf=inputlist)
    #C2_tot = ana.confs_mult(C2,-1)
    C57 = np.zeros((len(inputlist),T,2))
    #ana.write_data_ascii(C2_tot,datadir+'pi_unit_opposite_p0.dat',conf=inputlist)
    if read_c4:
        print("C55")
        C4D = ana.read_confs(rawdir,Corrs[2],inputlist,T)
        C4C = ana.read_confs(rawdir,Corrs[3],inputlist,T)
        C55 = ana.confs_subtr(C4D,C4C)
        ana.write_data_ascii(C4D,datadir+'C4D.dat',conf=inputlist)
        ana.write_data_ascii(C4C,datadir+'C4C.dat',conf=inputlist)
        ana.write_data_ascii(C55,datadir+'C55_unit.dat',conf=inputlist)
        # loop over all contributions for gamma_j
        #for i in range(3):
        #    print("C57")
        #    C4D = ana.read_confs(rawdir,Corrs[i*2+2],inputlist,T)
        #    C4C = ana.read_confs(rawdir,Corrs[i*2+3],inputlist,T)
        #    print("Read in done")
        #    # subtract crossed from direct diagram
        #    C57 = ana.confs_add(C57,ana.confs_subtr(C4D,C4C))
        #    print("Writing to: %s..." % datadir)
        #    #ana.write_data_ascii(C2,datadir+'pi_charged_p0.dat')
        #    # write out the sum over j of C^j_{57}D-C^j_{57}C 
        #ana.write_data_ascii(C57,datadir+'C57.dat',conf=inputlist)
    #    ana.write_data_ascii(C4D,datadir+'C4D.dat')
    #    ana.write_data_ascii(C4C,datadir+'C4C.dat')
    print("Finished")


  
if __name__ == "__main__":
    main()

