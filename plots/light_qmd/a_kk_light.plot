set title "K-K scattering length A-Ensembles"
set xlabel "M_k/f_k"
set ylabel "M_K a_kk"

file='a_kk_light_qmd.dat'

set terminal postscript solid colour landscape lw 1.5
set output 'mk_akk.eps'
set xrange [3.0:3.5]
set yrange [-0.7:-0.2]
set grid

plot file i 0 u ($10/$8):22:23 w yerr t 'amu_s = 0.0185',\
     file i 1 u ($10/$8):22:23 w yerr t 'amu_s = 0.0225',\
     file i 2 u ($10/$8):22:23 w yerr t 'amu_s = 0.02464'

#replot file i 0 u ($10/$8):($4*$12) t 'amu_s = 0.0185',\
#       file i 1 u ($10/$8):($4*$12) t 'amu_s = 0.0225',\
#       file i 2 u ($10/$8):($4*$12) t 'amu_s = 0.02464'

plot file i 0 u ($10/$8):22:23 w yerr t 'amu_s = 0.0185'
plot file i 1 u ($10/$8):22:23 w yerr t 'amu_s = 0.0225'
plot file i 2 u ($10/$8):22:23 w yerr t 'amu_s = 0.02464'


