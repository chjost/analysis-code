set title "Scattering length A-, B- and D-Ensembles"
#set xlabel "(r_0 M_K)^2" 
set xlabel "(r_0 M_Pi)^2" 
set ylabel "M_Ka_0"
set yrange [-0.44:-0.28]
set grid
r0_A2=5.31**2
r0_B2=5.77**2
r0_D2=7.60**2
set style line 5 lc rgb "red" pt 15
set style line 1 lc rgb "red" pt 17 
set style line 2 lc rgb "blue" pt 22
set style line 3 lc rgb "dark-green" pt 20
set style line 4 lc rgb "red" pt 18
#plot systematical error on top of statistical first (they are larger)
plot './kk_beta_mpi.dat' i 0 u ($2**2)*r0_A2:5:($5-$6-$7):($5+$6+$8) ls 5 w errorbars notitle,\
'./kk_beta_mpi.dat' i 1 u ($2**2)*r0_A2:5:($5-$6-$7):($5+$6+$8) w errorbars ls 1 notitle, \
'./kk_beta_mpi.dat' i 2 u ($2**2)*r0_A2:5:($5-$6-$7):($5+$6+$8) w errorbars ls 4 notitle, \
'./kk_beta_mpi.dat' i 3 u ($2**2)*r0_B2:5:($5-$6-$7):($5+$6+$8) w errorbars ls 2 notitle, \
'./kk_beta_mpi.dat' i 4 u ($2**2)*r0_D2:5:($5-$6-$7):($5+$6+$8) w errorbars ls 3 notitle

replot './kk_beta_mpi.dat' i 0 u ($2**2)*r0_A2:5:6 w yerr ls 5 t 'b1.90, mu_s = 0.0185', \
'./kk_beta_mpi.dat' i 1 u ($2**2)*r0_A2:5:6 w yerr ls 1 t 'b1.90, mu_s = 0.0225', \
'./kk_beta_mpi.dat' i 2 u ($2**2)*r0_A2:5:6 w yerr ls 4 t 'b1.90, mu_s = 0.02464', \
'./kk_beta_mpi.dat' i 3 u ($2**2)*r0_B2:5:6 w yerr ls 2 t 'b1.95, mu_s = 0.01860', \
'./kk_beta_mpi.dat' i 4 u ($2**2)*r0_D2:5:6 w yerr ls 3 t 'b2.10, mu_s = 0.01500'

set terminal pdfcairo enhanced solid color size 7,5 lw 2 font ",16"
set out './kk_scat_overview_mpi.pdf'
replot

#plot systematical error on top of statistical first (they are larger)
#plot './kk_beta_all.dat' i 0 u ($2**2)*r0_A2:14:($14-$15-$16):($14+$15+$17) ls 5 w errorbars notitle
#replot './kk_beta_all.dat' i 1 u ($2**2)*r0_A2:14:($14-$15-$16):($14+$15+$17) w errorbars ls 1 notitle
#replot './kk_beta_all.dat' i 2 u ($2**2)*r0_A2:14:($14-$15-$16):($14+$15+$17) w errorbars ls 4 notitle
#replot './kk_beta_all.dat' i 3 u ($2**2)*r0_B2:14:($14-$15-$16):($14+$15+$17) w errorbars ls 2 notitle
#replot './kk_beta_all.dat' i 4 u ($2**2)*r0_D2:14:($14-$15-$16):($14+$15+$17) w errorbars ls 3 notitle
#
#replot './kk_beta_all.dat' i 0 u ($2**2)*r0_A2:14:15 w yerr ls 5 t 'b1.90, mu_s = 0.0185'
#replot './kk_beta_all.dat' i 1 u ($2**2)*r0_A2:14:15 w yerr ls 1 t 'b1.90, mu_s = 0.0225'
#replot './kk_beta_all.dat' i 2 u ($2**2)*r0_A2:14:15 w yerr ls 4 t 'b1.90, mu_s = 0.02464'
#replot './kk_beta_all.dat' i 3 u ($2**2)*r0_B2:14:15 w yerr ls 2 t 'b1.95, mu_s = 0.01860'
#replot './kk_beta_all.dat' i 4 u ($2**2)*r0_D2:14:15 w yerr ls 3 t 'b2.10, mu_s = 0.01500'
#
#set terminal pdfcairo enhanced solid color size 7,5 lw 2 font ",16"
#set out './kk_scat_overview.pdf'
#replot
#
#plot './kk_beta_all.dat' i 0 u ($2**2)*r0_A2:14:($14-$15-$16):($14+$15+$17) ls 5 w errorbars notitle, \
#'./kk_beta_all.dat' i 3 u ($2**2)*r0_B2:14:($14-$15-$16):($14+$15+$17) w errorbars ls 2 notitle, \
#'./kk_beta_all.dat' i 4 u ($2**2)*r0_D2:14:($14-$15-$16):($14+$15+$17) w errorbars ls 3 notitle, \
#'./kk_beta_all.dat' i 0 u ($2**2)*r0_A2:14:15 w yerr ls 5 t 'b1.90, mu_s = 0.0185', \
#'./kk_beta_all.dat' i 3 u ($2**2)*r0_B2:14:15 w yerr ls 2 t 'b1.95, mu_s = 0.01860', \
#'./kk_beta_all.dat' i 4 u ($2**2)*r0_D2:14:15 w yerr ls 3 t 'b2.10, mu_s = 0.01500'
#plot './kk_beta_all.dat' i 1 u ($2**2)*r0_A2:14:($14-$15-$16):($14+$15+$17) ls 1 w errorbars notitle, \
#'./kk_beta_all.dat' i 3 u ($2**2)*r0_B2:14:($14-$15-$16):($14+$15+$17) w errorbars ls 2 notitle, \
#'./kk_beta_all.dat' i 4 u ($2**2)*r0_D2:14:($14-$15-$16):($14+$15+$17) w errorbars ls 3 notitle, \
#'./kk_beta_all.dat' i 1 u ($2**2)*r0_A2:14:15 w yerr ls 1 t 'b1.90, mu_s = 0.0225', \
#'./kk_beta_all.dat' i 3 u ($2**2)*r0_B2:14:15 w yerr ls 2 t 'b1.95, mu_s = 0.01860', \
#'./kk_beta_all.dat' i 4 u ($2**2)*r0_D2:14:15 w yerr ls 3 t 'b2.10, mu_s = 0.01500'
#plot './kk_beta_all.dat' i 2 u ($2**2)*r0_A2:14:($14-$15-$16):($14+$15+$17) ls 4 w errorbars notitle, \
#'./kk_beta_all.dat' i 3 u ($2**2)*r0_B2:14:($14-$15-$16):($14+$15+$17) w errorbars ls 2 notitle, \
#'./kk_beta_all.dat' i 4 u ($2**2)*r0_D2:14:($14-$15-$16):($14+$15+$17) w errorbars ls 3 notitle, \
#'./kk_beta_all.dat' i 2 u ($2**2)*r0_A2:14:15 w yerr ls 4 t 'b1.90, mu_s = 0.02464', \
#'./kk_beta_all.dat' i 3 u ($2**2)*r0_B2:14:15 w yerr ls 2 t 'b1.95, mu_s = 0.01860', \
#'./kk_beta_all.dat' i 4 u ($2**2)*r0_D2:14:15 w yerr ls 3 t 'b2.10, mu_s = 0.01500'
