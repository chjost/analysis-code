#!/usr/bin/python

################################################################################
# Author: Christopher Helmes helmes@hiskp.uni-bonn.de
# 
# Module interfacing with the correlation functions of Markus Werner's
# contraction code. All correlation functions for a range of configurations in
# one file are summed and written out in Liuming's format
#
#
#
#
#
#
################################################################################

import subprocess
import numpy as np

import input_output as io

# Wrapper for the lime_get_record program
# args is argumentlist
def extract_recs(prefix="C2_pi+-_conf",start_cfg=1870,delta_cfg=6, nb_cfg=8,
    T=48, rec=3, n_corrs=1):
  # In the given file format the correlation functions are written out as the 3
  # records of lime messages. Data begins at message 2
  # intermediate name
  tmp_name = "rec.tmp"
  # Array for all correlation functions
  single=np.zeros((nb_cfg,), '48c8')
  for cfg in range(start_cfg, start_cfg+delta_cfg*nb_cfg, delta_cfg):
    in_name = prefix + str(cfg) + '.dat'
    # Correlation functions from one configuration
    corrs_one_conf = np.zeros((n_corrs,), '48c8')
    #print(corrs_one_conf)
    for msg in range(2,2+n_corrs):
      out_name=tmp_name +str(cfg)+'.dat'
      #print(in_name, out_name)
      subprocess.check_call(["lime_extract_record", in_name, str(msg), str(rec),
        out_name])
      # append content of tmp name to array
      corrs_one_conf[msg-2] = io.extract_bin_corr_fct(tmp_name, cfg,
          1, 1, 48, 0)
      # temporary files are not needed any more
      subprocess.check_call(["rm",out_name])
    single[(cfg-start_cfg)/delta_cfg] = np.sum(corrs_one_conf, axis=0)
  return single

def main():
  T=48
  nb_cfg=8
  op_sum=extract_recs(n_corrs=9)
  corr_cfg=np.reshape(op_sum,op_sum.shape[0]*op_sum.shape[1])
  print(corr_cfg[0:47])
  io.write_corr_fct(corr_cfg,"/hiskp2/helmes/contractions/A40.20/merged/corrs.dat",T,nb_cfg,timesorted=False)
# make this script importable, according to the Google Python Style Guide                                                          
if __name__ == '__main__':                                                                                                         
  main()
