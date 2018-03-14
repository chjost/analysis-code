#!/bin/bash
# e is one of '', '_fp' or '_div' corresponding to E1, E2 and E3
e='_final'
meth=( A B )
zp=( 1 2 )
epik=( E1 E2 E3 )
for m in "${meth[@]}"; do
    for z in "${zp[@]}"; do
        infile="../ini/pi_K/I_32${e}/chiral_analysis_mua0_zp${z}.ini"
        sed  -i "s/.*epik_meth.*/epik_meth=${e_meth}/" ${infile}
        ./pik_I32_nlo_chipt ${infile} ${m} \
        > ./logfiles/pik_I32_nlo_chipt_${m}${z}_${emeth}${e}.log
        ./pik_I32_gamma ${infile} ${m} \
        > ./logfiles/pik_I32_gamma_${m}${z}_${emeth}${e}.log
    done
done

