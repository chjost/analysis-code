set title "Scattering length A-, B- and D-Ensembles"
set xlabel "(r_0 M_K)^2" 
set ylabel "M_Ka_0"
set grid
r0_A2=5.31**2
r0_B2=5.77**2
r0_D2=7.60**2
set style line 1 lc rgb "red" pt 17 
set style line 2 lc rgb "blue" pt 22
set style line 3 lc rgb "dark-green" pt 20
set style line 4 lc rgb "red" pt 18
#plot statistical errors first (they are larger)
plot './kk_beta_all.dat' i 0 u ($2**2)*r0_A2:14:($14-$15-$17):($14+$15+$16) ls 1 w errorbars notitle
replot './kk_beta_all.dat' i 1 u ($2**2)*r0_A2:14:($14-$15-$17):($14+$15+$16) w errorbars ls 4 notitle
replot './kk_beta_all.dat' i 2 u ($2**2)*r0_B2:14:($14-$15-$17):($14+$15+$16) w errorbars ls 2 notitle
replot './kk_beta_all.dat' i 3 u ($2**2)*r0_D2:14:($14-$15-$17):($14+$15+$16) w errorbars ls 3 notitle

replot './kk_beta_all.dat' i 0 u ($2**2)*r0_A2:14:15 w yerr ls 1 t 'b1.90, mu_s = 0.02250'
replot './kk_beta_all.dat' i 1 u ($2**2)*r0_A2:14:15 w yerr ls 4 t 'b1.90, mu_s = 0.02464'
replot './kk_beta_all.dat' i 2 u ($2**2)*r0_B2:14:15 w yerr ls 2 t 'b1.95, mu_s = 0.01860'
replot './kk_beta_all.dat' i 3 u ($2**2)*r0_D2:14:15 w yerr ls 3 t 'b2.10, mu_s = 0.01500'

set terminal pdfcairo enhanced solid color size 7,5 lw 2 font ",16"
set out './kk_scat_overview.pdf'
replot

