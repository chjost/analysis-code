#!/bin/bash
meth=( A B )
zp=( zp1 zp2 )
for m in "${meth[@]}"; do
    for z in "${zp[@]}"; do
        infile="../ini/pi_K/I_32/chiral_analysis_mua0_${z}.ini"
        ./fix_ms_${m} ${infile} > fix_ms_${m}_${z}.log
    done
done
           
