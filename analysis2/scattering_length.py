"""
Functions related to the scattering length.
"""

import numpy as np

#def calculate_scat_len(mass, massweight, energy, energyweight, L=24,
#        isdependend=True, isratio=False):
#    """Calculate the scattering length with the Luescher Formula.
#    """
#    cut = 1e-14
#    nsam = mass.shape[0]
#    # Constants for the Luescher Function
#    c = [-2.837297, 6.375183, -8.311951]
#    # prefactor of the equation
#    pre = -4.*np.pi / (mass * float(L*L*L))
#    needed = np.zeros((nsam,))
#    # loop over fitranges of self
#    for i in range(energy.shape[-1]):
#        # loop over fitranges of mass
#        for j in range(mass.shape[-1]):
#            result = np.zeros((nsam,))
#            weight = np.zeros((nsam,))
#            # check if the weight is smaller than cut-off, if so
#            # don't calculate
#            if isratio or isdependend:
#                weight = np.full(nsam, massweight[j] * energyweight[j,i])
#            else:
#                weight = np.full(nsam, massweight[j] * energyweight[i])
#                #weight = massweight[j] * energyweight[i]
#            if False:
#            #if weight[0] < cut or weight[0] > (1. - cut):
#                result = np.full(nsam, np.nan)
#                print("cut away")
#            else:
#                # loop over samples
#                for b in range(nsam):
#                    p = np.asarray([pre[b,j]*c[1]/float(L*L),
#                        pre[b,j]*c[0]/float(L), pre[b,j], 0.])
#                    if isratio:
#                        p[3] = -1. * energy[b,j,i]
#                      
#                    else:
#                        if isdependend:
#                            p[3] = -1. * (energy[b,j,i]-2*mass[b,j])
#                        else:
#                            p[3] = -1. * (energy[b,i]-2*mass[b,j])
#                    # find the roots of the polynomial
#                    root = np.roots(p)
#                    # sort by absolute value of imaginary part
#                    ind_root = np.argsort(np.fabs(root.imag))
#                    # the first entry is the wanted
#                    result[b] = root[ind_root][0].real
#            yield (0, 0, j, i), result, needed, weight
#
#if __name__ == "main":
#    pass

# This is code doubling to try out what happens, if effective range is taken
# into account
def calculate_scat_len(mass, massweight, energy, energyweight, L=24,
        isdependend=True, isratio=False, rf_est=None):
    """Calculate the scattering length with the Luescher Formula.
    """
    cut = 1e-14
    nsam = mass.shape[0]
    # Constants for the Luescher Function
    c = [-2.837297, 6.375183, -8.311951]
    # prefactor of the equation
    pre = -4.*np.pi / (mass * float(L*L*L))
    needed = np.zeros((nsam,))
    # loop over fitranges of self
    for i in range(energy.shape[-1]):
        # loop over fitranges of mass
        for j in range(mass.shape[-1]):
            result = np.zeros((nsam,))
            weight = np.zeros((nsam,))
            # check if the weight is smaller than cut-off, if so
            # don't calculate
            if isratio or isdependend:
                weight = np.full(nsam, massweight[j] * energyweight[j,i])
            else:
                weight = np.full(nsam, massweight[j] * energyweight[i])
                #weight = massweight[j] * energyweight[i]
            if False:
            #if weight[0] < cut or weight[0] > (1. - cut):
                result = np.full(nsam, np.nan)
                print("cut away")
            else:
                # loop over samples
                for b in range(nsam):
                    # Use estimate of rf, calculate a0 up to O(1/L^6)
                    if rf_est is not None:
                        #print(rf_est[b])
                        p = np.asarray([0., pre[b,j]*c[1]/float(L*L),
                            pre[b,j]*c[0]/float(L), pre[b,j], 0.])
                        p[0] = pre[b,j]/float(L*L*L)*( c[2] + 2*np.pi*rf_est[b])
                        # Neglect effective range estimate
                        #p[0] = pre[b,j]/float(L*L*L)*c[2]
                        if isratio:
                            p[4] = -1. * energy[b,j,i]
                          
                        else:
                            if isdependend:
                                p[4] = -1. * (energy[b,j,i]-2*mass[b,j])
                            else:
                                p[4] = -1. * (energy[b,i]-2*mass[b,j])
                    # No estimate of rf
                    else:
                        p = np.asarray([pre[b,j]*c[1]/float(L*L),
                            pre[b,j]*c[0]/float(L), pre[b,j], 0.])
                        if isratio:
                            p[3] = -1. * energy[b,j,i]
                          
                        else:
                            if isdependend:
                                p[3] = -1. * (energy[b,j,i]-2*mass[b,j])
                            else:
                                p[3] = -1. * (energy[b,i]-2*mass[b,j])
                    # find the roots of the polynomial
                    root = np.roots(p)
                    # sort by absolute value of imaginary part (not needed
                    # anymore?)
                    #ind_root = np.argsort(np.fabs(root.imag))
                    # the first entry is the wanted
                    #result[b] = root[ind_root][0].real
                    # instead of taking the last entry throw out all
                    # complex roots and
                    mask = np.isreal(root)
                    real_roots = root[mask]
                    real_ind = np.argsort(np.fabs(real_roots.real))
                    result[b] = real_roots[real_ind][0].real
            yield (0, 0, j, i), result, needed, weight

if __name__ == "main":
    pass

