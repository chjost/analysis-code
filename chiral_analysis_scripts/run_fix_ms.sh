#!/bin/bash
# e is one of '', '_fp' or '_div' corresponding to E1, E2 and E3
# e is '_final'
e='_final'
meth=( A B )
zp=( 1 2 )
epik=( E1 E2 E3 )
for m in "${meth[@]}"; do
    for z in "${zp[@]}"; do
        for e_meth in "${epik[@]}"; do
            infile="../ini/pi_K/I_32${e}/chiral_analysis_mua0_zp${z}.ini"
            sed  -i "s/.*epik_meth.*/epik_meth=${e_meth}/" ${infile}
            ./fix_ms_${m} ${infile} > ./logfiles/fix_ms_${m}_${z}_${e_meth}${e}.log
            #./fix_ms_${m} ${infile}
        done
    done
done
           
