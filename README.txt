The branch kk_new contains tools for a more structured analysis of the Kaon Kaon
scattering length at P = 0. The parameters are controlled by the .ini files. The
usual toolchain is:
(0) corr_test: Used to look at timeslice histories and occasionally omit
configurations from the analysis
1) fit_all: Fits single pion energy and energy shift to ratio
2) kk_scat: calculates the scattering length from the single energies and the
energy shift following Lueschers scattering formula
3) m_a0: This builds the product between single kaon mass and scattering length
4) match_qm: Matches the strange quark mass to a given observable (typically a
meson mass)
