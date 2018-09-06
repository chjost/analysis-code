#!/bin/bash
# e is one of '', '_fp' or '_div' corresponding to E1, E2 and E3
e='_cov_false'
meth=( A )
zp=( 1 )
epik=( 1 2 )
for m in "${meth[@]}"; do
    for z in "${zp[@]}"; do
        for e_meth in ${epik[@]}; do
            infile="../ini/pi_K/I_32${e}/chiral_analysis_mua0.ini"
            ./pik_I32_nlo_chipt_plots.py --infile ${infile} --msfix ${m}\
              --zp ${z} --epik ${e_meth}\
            > ./logfiles/pik_I32_nlo_chipt_plots_${m}${z}_${e_meth}${e}.log
            ./pik_I32_gamma_plots.py --infile ${infile} --msfix ${m}\
              --zp ${z} --epik ${e_meth}\
            > ./logfiles/pik_I32_gamma_plots_${m}${z}_${e_meth}${e}.log
            #./pik_I32_nlo_chipt_plots_deviation.py --infile ${infile} --msfix ${m}\
            #  --zp ${z} --epik ${e_meth}\
            #> ./logfiles/pik_I32_gamma_plots_${m}${z}_${e_meth}${e}.log
            
        done
    done
done


