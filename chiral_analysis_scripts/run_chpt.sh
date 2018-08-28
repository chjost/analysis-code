#!/bin/bash
# e is one of '', '_fp' or '_div' corresponding to E1, E2 and E3
e='_cov_false'
meth=( A )
zp=( 1 )
epik=( E1 E2 )
for m in "${meth[@]}"; do
    for z in "${zp[@]}"; do
        for e_meth in ${epik[@]}; do
            infile="../ini/pi_K/I_32${e}/chiral_analysis_mua0.ini"
            sed  -i "s/.*epik_meth.*/epik_meth=${e_meth}/" ${infile}
            ./pik_I32_nlo_chipt.py ${infile} ${m} \
            > ./logfiles/pik_I32_nlo_chipt_${m}${z}_${e_meth}${e}.log
            ./pik_I32_gamma.py ${infile} ${m} \
            > ./logfiles/pik_I32_gamma_${m}${z}_${e_meth}${e}.log
        done
    done
done

