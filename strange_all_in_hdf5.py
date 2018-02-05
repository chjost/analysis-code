#!/usr/bin/python
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
    c2_pi = True 
    c2_ss = False
    # parse the input file
    if len(sys.argv) < 2:
        ens = ana.LatticeEnsemble.parse("charged.ini")
    else:
        ens = ana.LatticeEnsemble.parse(sys.argv[1])
    # get data from input file
    lat = ens.get_data("name")
    T = ens.get_data("T")
    rawdir = ens.get_data("rawdir_kk")
    raw_pi = ens.get_data("rawdir_pi")
    raw_ss = ens.get_data("rawdir_ss")
    datadir = ens.get_data("datadir") 
    rawstrange = ens.get_data("raw_a")
    strange = ens.get_data("strangea")
    # TODO: Place that in the ensemble class
    if len(sys.argv) < 2:
      Corrs = ana.inputnames('charged.ini',['C20', 'C40C', 'C40D'],h5=True)
    else:
      Corrs = ana.inputnames(sys.argv[1],['c1','c2', 'c3'],h5=True)
      #Corrs = ana.inputnames(sys.argv[1],['c0'])
      Corrs_pi = ana.inputnames(sys.argv[1],['c0'],h5=True)
      Corrs_ss = ana.inputnames(sys.argv[1],['c6'],h5=True)
    # os.path.join treats preceding slashes as new paths
    print(Corrs)
    print(Corrs_pi)
    print(Corrs_ss)
    corrpaths = [os.path.join(rawdir,mu_s,'data/') for mu_s in rawstrange]
    corrpaths_pi = [os.path.join(raw_pi,mu_s,'data/') for mu_s in rawstrange]
    #corrpaths_pi = [raw_pi]
    corrpaths_ss = [os.path.join(raw_ss,mu_s,'data/') for mu_s in rawstrange]
    datapaths = [os.path.join(datadir,mu_s,'') for mu_s in strange]
    # get all available correlators in three lists
    # TODO: Filter out everything not starting with 'conf'
    configs_coll = [os.listdir(cp) for cp in corrpaths]
    # find all 4digit occurences in each filename and convert them to a list
    conf_feed = [set([map(int,re.findall(r'\d{4}', c))[0] for c in
                 os.listdir(strange_path)]) for strange_path in corrpaths ] 
    # get intersections of config sets
    conf_feed = conf_feed[0] & conf_feed[1] & conf_feed[2]

    conf_feed_list = sorted(list(conf_feed),key=int)
    conf_feed_stringlist = ["%04d"%c for c in conf_feed_list]
    print("conf_feed has length: %r" %len(conf_feed_stringlist))
    print(conf_feed_stringlist)
    # Read in kaon data
    for s in zip(corrpaths,datapaths):
      print(s)
      # copy common subset of configurations to appropriate target directory:
      # Read in correlators
      print("Reading Correlation functions from %s..." % s[0])
      print("C2")
      C2 = ana.read_confs(s[0]+"/C2+_cnfg",Corrs[0],conf_feed_stringlist,
                          T,h5=True,verb=True)
      print("C4")
      C4D = ana.read_confs(s[0]+"/C4+D_cnfg" ,Corrs[1],conf_feed_stringlist,
                           T,h5=True)
      C4C = ana.read_confs(s[0]+"/C4+C_cnfg",Corrs[2],conf_feed_stringlist,
                           T,h5=True)
      print("Read in done")
      # subtract crossed from direct diagram
      C4_tot = ana.confs_subtr(C4D,C4C)
      #C4_tot = ana.confs_mult(C4_tot,2)
      print("Writing to: %s..." % s[1])
      # print first 10 lines of correlation function
      #ana.write_data_ascii(C2,s[1]+'pi_charged_p0.dat')
      # generate config numbers they are arrays of shape (nconf,T)
      config=np.asarray([np.repeat(int(c),T) for c in conf_feed_stringlist])
      print(config.shape)
      ana.write_data_ascii(C2,s[1]+'k_charged_p0_outlier.dat',conf=config)
      #ana.write_data_ascii(C2,s[1]+'pi_charged_p0_outlier.dat',conf=conf_feed)
      ana.write_data_ascii(C4_tot,s[1]+'pik_charged_A1_TP0_00_outlier.dat',conf=config)
      ana.write_data_ascii(C4D,s[1]+'C4D.dat',conf=config)
      ana.write_data_ascii(C4C,s[1]+'C4C.dat',conf=config)
      del C2
     
    # Read in pion data
    if c2_pi:
        p = raw_pi 
        print(p)
        # copy common subset of configurations to appropriate target directory:
        # Read in correlators
        print("Reading Pion Correlation functions from %s..." % corrpaths_pi)
        print("C2")
        # Correlators for pion are in ASCII format
        #C2 = np.array(len(conf_feed,T,3)) 
        #for i,c in enumerate(conf_feed):
        #  inname = corrpaths_pi[0]+'/pi_corr_p0.conf%04d.dat'%c 
        #  _C2 = ana.read_data_ascii(inname,column=(1,2),noheader=True,skip=1)
        #  _numconf = np.tile(c,T)
        #  _Cfull = np.vstack(_C2,_numconf)
        #  C2[i] = _Cfull
        C2 = ana.read_confs(corrpaths_pi[0]+"/C2+_cnfg",Corrs_pi[0],conf_feed_stringlist,T,h5=True)
        print("Read in done")
        save_pi = datadir+'pi/'
        print("Writing to: %s..." % save_pi)
        ana.write_data_ascii(C2,save_pi +'pi_charged_p0_outlier.dat',conf=config)
    
    # read ss data
    if c2_ss:
        for s in zip(corrpaths_ss,datapaths):
          print(s)
          # copy common subset of configurations to appropriate target directory:
          # Read in correlators
          print("Reading Correlation functions from %s..." % s[0])
          print("C2")
          C2 = ana.read_confs(s[0],Corrs_ss[0],conf_feed_stringlist,T,h5=True)
          print("Read in done")
          print("Writing to: %s..." % s[1])
          ana.write_data_ascii(C2,s[1]+'ss_charged_p0_outlier.dat',conf=config)
     
    print("Finished")

if __name__ == "__main__":
    main()

