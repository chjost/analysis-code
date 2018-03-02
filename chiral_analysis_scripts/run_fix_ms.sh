#!/bin/bash
# e is one of '', '_fp' or '_div' corresponding to E1, E2 and E3
e='_div'
meth=( A B )
zp=( 1 2 )
for m in "${meth[@]}"; do
    for z in "${zp[@]}"; do
        infile="../ini/pi_K/I_32${e}/chiral_analysis_mua0_zp${z}.ini"
        ./fix_ms_${m} ${infile} > ./logfiles/fix_ms_${m}_${z}${e}.log
    done
done
           
