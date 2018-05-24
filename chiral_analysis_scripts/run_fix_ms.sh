#!/bin/bash
# e is one of '', '_fp' or '_div' corresponding to E1, E2 and E3
# e is '_final'
e='_blocked'
# We have two methods for fixing the strange quark mass, labelled A and B
meth=( B )
# Furthermore there are two values for Z_P around (choose 1, 2)
zp=( 1 )
# Lastly we employ three methods for extracting E_piK ( E1 E2 E3 )
epik=( E1 E3 )
for m in "${meth[@]}"; do
    for z in "${zp[@]}"; do
        for e_meth in "${epik[@]}"; do
            infile="../ini/pi_K/I_32${e}/chiral_analysis_mua0_zp${z}.ini"
            sed  -i "s/.*epik_meth.*/epik_meth=${e_meth}/" ${infile}
            ./fix_ms_${m} ${infile} > ./logfiles/fix_ms_${m}_${z}_${e_meth}${e}.log
        done
    done
done
           
