import numpy as np

# Logarithms have the form 1/(16*pi**2)*m**2/f**2*np.log(m**2/mu**2)
def mass_log(mass,f,mu=None):
    """ Logarithms for chpt formulae 

    The input data can be all in lattice units or all in physical units. No
    mixing, since only ratios enter here. If the renormalization scale is not
    given it is automatically set to the chiral decayconstant f 
    Arguments
    ---------
    mass: 1d-array, bootstrapsamples of the corresponding meson mass
    f: 1d-array, chiral decayconstant (?)
    mu: 1d-array, renormalization scale, set to f if not given explicitly

    Returns
    -------
    l_mass: 1d-array, of the calculated values of the chiral logarithm
    """
    if mu is None:
        mu = f
    l_mass = 1./(16.*np.pi**2)*(mass/f)**2*np.log((mass/mu)**2)
    return l_mass
