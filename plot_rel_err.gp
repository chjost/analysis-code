set title "relative error delta E A-, B- and D-Ensembles"
set xlabel "(r_0 M_K)^2" 
set ylabel "relative error"
set grid
r0_A2=5.31**2
r0_B2=5.77**2
r0_D2=7.60**2
set style line 1 lc rgb "red" pt 17 
set style line 2 lc rgb "blue" pt 22
set style line 3 lc rgb "dark-green" pt 20
set style line 4 lc rgb "red" pt 18
#plot statistical errors first (they are larger)
#set offset 1,1,1,1
set terminal pdfcairo enhanced solid color size 7,5 lw 2 font ",16"
#set out './kk_delta_e_rel_err.pdf'
#set ylabel "delta dE rel"
#plot './kk_beta_all.dat' i 0 u ($2**2)*r0_A2:($7/$6):xticlabels(18) ls 1 t 'b1.90, mu_s = 0.02250', \
#     './kk_beta_all.dat' i 1 u ($2**2)*r0_A2:($7/$6):xticlabels(18) ls 4 t 'b1.90, mu_s = 0.02464', \
#     './kk_beta_all.dat' i 2 u ($2**2)*r0_B2:($7/$6):xticlabels(18) ls 2 t 'b1.95, mu_s = 0.01860', \
#     './kk_beta_all.dat' i 3 u ($2**2)*r0_D2:($7/$6):xticlabels(18) ls 3 t 'b2.10, mu_s = 0.01500'
#
#set out './kk_a0_rel_err.pdf'
#set ylabel "delta a_0 rel"
#set key right center
#plot './kk_beta_all.dat' i 0 u ($2**2)*r0_A2:($11/$10):xticlabels(18) ls 1 t 'b1.90, mu_s = 0.02250', \
#     './kk_beta_all.dat' i 1 u ($2**2)*r0_A2:($11/$10) ls 4 t 'b1.90, mu_s = 0.02464', \
#     './kk_beta_all.dat' i 2 u ($2**2)*r0_B2:($11/$10) ls 2 t 'b1.95, mu_s = 0.01860', \
#     './kk_beta_all.dat' i 3 u ($2**2)*r0_D2:($11/$10) ls 3 t 'b2.10, mu_s = 0.01500'
#
set out './kk_dE_by_a0_rel.pdf'
set ylabel "delta dE/delta a0"
plot './kk_beta_all.dat' i 0 u ($2**2)*r0_A2:(($7/$6)/($11/$10)):18 with labels ls 1 t 'b1.90, mu_s = 0.02250', \
     './kk_beta_all.dat' i 1 u ($2**2)*r0_A2:(($7/$6)/($11/$10)):18 with labels ls 4 t 'b1.90, mu_s = 0.02464', \
     './kk_beta_all.dat' i 2 u ($2**2)*r0_B2:(($7/$6)/($11/$10)):18 with labels ls 2 t 'b1.95, mu_s = 0.01860', \
     './kk_beta_all.dat' i 3 u ($2**2)*r0_D2:(($7/$6)/($11/$10)):18 with labels ls 3 t 'b2.10, mu_s = 0.01500'


