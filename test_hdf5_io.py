#!/usr/bin/python
import sys
import numpy as np
import os
import re
import analysis2 as ana


def main():
    ens = ana.LatticeEnsemble.parse(sys.argv[1])
    print(sys.argv[1])
    path_2pt="/hiskp4/helmes/contractions/pi_k/I_32/A40.20/strange_1850/data/C2+_cnfg"
    path_4pt_d="/hiskp4/helmes/contractions/pi_k/I_32/A40.20/strange_1850/data/C4+D_cnfg"
    path_4pt_c="/hiskp4/helmes/contractions/pi_k/I_32/A40.20/strange_1850/data/C4+C_cnfg"
    Corrs = ana.inputnames(sys.argv[1],['C2+','C2+','C4+D','C4+C'],h5=True)
    print(Corrs)
    #C2_one = ana.read_confs(path,Corrs[0],["0202"],64,h5=True,verb=True)
    #C2_two = ana.read_confs(path,Corrs[1],["0202"],64,h5=True,verb=True)
    #C2_three = ana.read_confs(path,Corrs[2],["0202"],64,h5=True,verb=True)
    #C2_sum = confs_add(C2_one,C2_two)
    #C2_sum = confs_add(C2_sum,C2_three)
    #print(C2_sum)
    #ana.write_data_ascii(C2_sum,"./C2.dat",conf=np.full((1,48),1372))

    C2_pi = ana.read_confs(path_2pt,Corrs[0],["1372"],48,h5=True,verb=True)
    C2_k = ana.read_confs(path_2pt,Corrs[1],["1372"],48,h5=True,verb=True)
    C4d = ana.read_confs(path_4pt_d,Corrs[2],["1372"],48,h5=True,verb=True)
    C4c = ana.read_confs(path_4pt_c,Corrs[3],["1372"],48,h5=True,verb=True)



if __name__ == "__main__":
    main()

