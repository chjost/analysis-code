from chipt_basic_observables import *
from chipt_nlo import *
from pik_scat_len import *

"""Wrapper functions for fits and plots"""
def calc_x_plot(x):
    """ Function that calculates reduced mass divided by f_pi from mk,mpi and
    fpi"""
    xplot=reduced_mass(x[:,0],x[:,1])/x[:,2]
    #xplot=np.asarray((x[:,1]/x[:,0]))
    return xplot

def calc_x_plot_cont(x):
    """ Function that calculates reduced mass divided by f_pi from ml, ms, b0 and
    l4"""
    mpi=np.sqrt(mpi_sq(x[:,0],b0=x[:,3]))
    mk=np.sqrt(mk_sq(x[:,0],ms=x[:,1],b0=x[:,3]))
    fpi=f_pi(x[:,0],f0=None,l4=x[:,2],b0=x[:,3])
    xplot=reduced_mass(mpi,mk)/fpi
    #xplot=mk/mpi
    return xplot
def err_func(p, x, y, error):
    # for each lattice spacing and prior determine the dot product of the error
    chi_a = y.A - mu_pik_a_32_fit(p,x.A)
    chi_b = y.B - mu_pik_a_32_fit(p,x.B)
    chi_d = y.D - mu_pik_a_32_fit(p,x.D)
    # TODO: find a runtime way to disable prior
    #if len(y._fields) > 3:
    try:
        chi_p = y.p - p[1]
        return np.dot(error,np.r_[chi_a,chi_b,chi_d,chi_p])
        #return np.dot(error,np.r_[chi_a,chi_b,chi_p])
    except:
        # and append them to a vector
        return np.dot(error,np.r_[chi_a,chi_b,chi_d])
        #return np.dot(error,np.r_[chi_a,chi_b])
    #return np.dot(error,np.r_[chi_a])

# TODO: This is a heck of code doubling think about how to organize that better    
def err_func_mpi(p, x, y, error):
    # for each lattice spacing and prior determine the dot product of the error
    chi_a = y.A - mu_pik_a_32_fit_mpi(p,x.A)
    chi_b = y.B - mu_pik_a_32_fit_mpi(p,x.B)
    chi_d = y.D - mu_pik_a_32_fit_mpi(p,x.D)
    # TODO: find a runtime way to disable prior
    if len(y._fields) > 3:
        chi_p = y.p - p[1]
        return np.dot(error,np.r_[chi_a,chi_b,chi_d,chi_p])
    else:
        # and append them to a vector
        return np.dot(error,np.r_[chi_a,chi_b,chi_d])
    #return np.dot(error,np.r_[chi_a])
def err_func_mk(p, x, y, error):
    # for each lattice spacing and prior determine the dot product of the error
    chi_a = y.A - mu_pik_a_32_fit_mk(p,x.A)
    chi_b = y.B - mu_pik_a_32_fit_mk(p,x.B)
    chi_d = y.D - mu_pik_a_32_fit_mk(p,x.D)
    if len(y._fields) > 3:
        chi_p = y.p - p[1]
        return np.dot(error,np.r_[chi_a,chi_b,chi_d,chi_p])
    else:
        # and append them to a vector
        return np.dot(error,np.r_[chi_a,chi_b,chi_d])
    #return np.dot(error,np.r_[chi_a])
def err_func_mlms(p, x, y, error):
    # for each lattice spacing and prior determine the dot product of the error
    chi_a = y.A - mu_pik_a_32_fit_mlms(p,x.A)
    chi_b = y.B - mu_pik_a_32_fit_mlms(p,x.B)
    chi_d = y.D - mu_pik_a_32_fit_mlms(p,x.D)
    # TODO: find a runtime way to disable prior
    if len(y._fields) > 3:
        chi_p = y.p - p[1]
        return np.dot(error,np.r_[chi_a,chi_b,chi_d,chi_p])
    else:
        # and append them to a vector
        return np.dot(error,np.r_[chi_a,chi_b,chi_d])
def err_func_mupik(p, x, y, error):
    # for each lattice spacing and prior determine the dot product of the error
    chi_a = y.A - mu_pik_a_32_fit_mupik(p,x.A)
    chi_b = y.B - mu_pik_a_32_fit_mupik(p,x.B)
    chi_d = y.D - mu_pik_a_32_fit_mupik(p,x.D)
    # TODO: find a runtime way to disable prior
    if len(y._fields) > 3:
        chi_p = y.p - p[1]
        return np.dot(error,np.r_[chi_a,chi_b,chi_d,chi_p])
    else:
        # and append them to a vector
        return np.dot(error,np.r_[chi_a,chi_b,chi_d])

def line(p,x):
    _res = p[0]-2.*x*p[1]
    return _res

def gamma_errfunc(p,x,y,error):
    chi_a = y.A - line(p,x.A)
    chi_b = y.B - line(p,x.B)
    chi_d = y.D - line(p,x.D)
    # p[0] is L_5
    try:
        chi_p = y.p - p[0]
        return np.dot(error,np.r_[chi_a,chi_b,chi_d,chi_p])
        #return np.dot(error,np.r_[chi_a,chi_b,chi_p])
    except:
        # and append them to a vector
        return np.dot(error,np.r_[chi_a,chi_b,chi_d])
        #return np.dot(error,np.r_[chi_a,chi_b])
def mu_pik_a_32_fit(p,x):
    """Wrapper for fitting the chipt formula of mu_pik_a_32 to the lattice data
    
    The chipt function for mu_pik a_32 gets evaluated and augmented with a
    possible lattice artefact. 

    Inputs
    ------
    p: 2d-array, array of shape (nboot,npar), the fit parameters, if npar > 2,
       last entry is trated as lattice artefact
    x: nd-array, measured input from lattice data (nboot,ninputs)

    Returns
    -------
    _res: 1d-array, samples of the variable mu_pik a_32
    """
    # The argument order for pik_I32_chipt_nlo is:
    # mpi, mk, fpi, p, lambda_x, meta
    _res = pik_I32_chipt_nlo(x[:,0],x[:,1],x[:,2], p,meta=x[:,4])
    # The third parameter gets treated as parameter for the lattice artefact
    #TODO: Parameter array has wrong shape can we change that?
    #if p.shape[0] > 2:
        # For the time being choose mpi**2 as lattice artefact
        #_res += p[2]*reduced_mass(x[:,0],x[:,1])**2
    return _res
def mu_pik_a_32_fit_mpi(p,x):
    """Wrapper for fitting the chipt formula of mu_pik_a_32 to the lattice data
    
    The chipt function for mu_pik a_32 gets evaluated and augmented with a
    possible lattice artefact. 

    Inputs
    ------
    p: 2d-array, array of shape (nboot,npar), the fit parameters, if npar > 2,
       last entry is trated as lattice artefact
    x: nd-array, measured input from lattice data (nboot,ninputs)

    Returns
    -------
    _res: 1d-array, samples of the variable mu_pik a_32
    """
    # The argument order for pik_I32_chipt_nlo is:
    # mpi, mk, fpi, p, lambda_x, meta
    _res = pik_I32_chipt_nlo(x[:,0],x[:,1],x[:,2], p,meta=x[:,4])
    # The third parameter gets treated as parameter for the lattice artefact
    #TODO: Parameter array has wrong shape can we change that?
    if p.shape[0] > 2:
        # For the time being choose mpi**2 as lattice artefact
        _res += p[2]*x[:,0]**2
    return _res
def mu_pik_a_32_fit_mk(p,x):
    """Wrapper for fitting the chipt formula of mu_pik_a_32 to the lattice data
    
    The chipt function for mu_pik a_32 gets evaluated and augmented with a
    possible lattice artefact. 

    Inputs
    ------
    p: 2d-array, array of shape (nboot,npar), the fit parameters, if npar > 2,
       last entry is trated as lattice artefact
    x: nd-array, measured input from lattice data (nboot,ninputs)

    Returns
    -------
    _res: 1d-array, samples of the variable mu_pik a_32
    """
    # The argument order for pik_I32_chipt_nlo is:
    # mpi, mk, fpi, p, lambda_x, meta
    _res = pik_I32_chipt_nlo(x[:,0],x[:,1],x[:,2], p,meta=x[:,4])
    # The third parameter gets treated as parameter for the lattice artefact
    #TODO: Parameter array has wrong shape can we change that?
    if p.shape[0] > 2:
        # For the time being choose mpi**2 as lattice artefact
        _res += p[2]*(x[:,1])**2
    return _res
def mu_pik_a_32_fit_mlms(p,x):
    """Wrapper for fitting the chipt formula of mu_pik_a_32 to the lattice data
    
    The chipt function for mu_pik a_32 gets evaluated and augmented with a
    possible lattice artefact. 

    Inputs
    ------
    p: 2d-array, array of shape (nboot,npar), the fit parameters, if npar > 2,
       last entry is trated as lattice artefact
    x: nd-array, measured input from lattice data (nboot,ninputs)

    Returns
    -------
    _res: 1d-array, samples of the variable mu_pik a_32
    """
    # The argument order for pik_I32_chipt_nlo is:
    # mpi, mk, fpi, p, lambda_x, meta
    _res = pik_I32_chipt_nlo(x[:,0],x[:,1],x[:,2], p,meta=x[:,4])
    # The third parameter gets treated as parameter for the lattice artefact
    #TODO: Parameter array has wrong shape can we change that?
    if p.shape[0] > 2:
        # For the time being choose mpi**2 as lattice artefact
        _res += p[2]*(0.5*x[:,0]**2+x[:,1]**2)
    return _res
def mu_pik_a_32_fit_mupik(p,x):
    """Wrapper for fitting the chipt formula of mu_pik_a_32 to the lattice data
    
    The chipt function for mu_pik a_32 gets evaluated and augmented with a
    possible lattice artefact. 

    Inputs
    ------
    p: 2d-array, array of shape (nboot,npar), the fit parameters, if npar > 2,
       last entry is trated as lattice artefact
    x: nd-array, measured input from lattice data (nboot,ninputs)

    Returns
    -------
    _res: 1d-array, samples of the variable mu_pik a_32
    """
    # The argument order for pik_I32_chipt_nlo is:
    # mpi, mk, fpi, p, lambda_x, meta
    _res = pik_I32_chipt_nlo(x[:,0],x[:,1],x[:,2], p,meta=x[:,4])
    # The third parameter gets treated as parameter for the lattice artefact
    #TODO: Parameter array has wrong shape can we change that?
    if p.shape[0] > 2:
        # For the time being choose mpi**2 as lattice artefact
        _res += p[2]*reduced_mass(x[:,0],x[:,1])**2
    return _res

# The plotting knows only one variable, the light quark mass. We therefore must
# calculate everything using at least LO formulae
def mu_pik_a_32_plot(p,x):
    """Evaluation of the fitted formula for plots

    This is necessary because mpi, mk, fpi and meta depend on each other so that
    we express everything in terms of the light quark mass. For this we need
    additional chipt constants (B_0, l_4, ...) and the fixed strange quark mass.
    They are given via x and constant.

    Inputs
    ------
    p: 2d-array, array of shape (npar,nboot), the fit parameters, if npar > 2,
       last entry is trated as lattice artefact
    x: nd-array, 0th row is the light quark mass, then necessary quantities for
       the chipt formulae coming into play. (ml,ms,b0,a*laq)

    Returns
    -------
    _res 1d-array, value of mu_pik a_32 for the plot
    """
    # Check if we have enough array entries
    if x.shape[1] < 3:
        print("\nIn mu_pik_a_32_plot: too less x-values:")
        print("want at least 3, have %d" %(x.shape[1]))
        _res = -1
    else:
      # inspect individual objects
        print("\nmua0_cont inputs:")
        #print("ml: %r" %x[:,0])
        #print("B0: %r" %x[:,2])
        #print("ms: %r" %x[:,1])
        #print("parameters: %r" %p)
        _mpi = np.sqrt(mpi_sq(x[:,0],x[:,2]))
        _mk = np.sqrt(mk_sq(x[:,0],x[:,1],x[:,2]))
        _fpi = f_pi(x[:,0],f0=None,l4=None,b0=x[:,2])
        print("mpi: %r" %_mpi[0])
        print("mk: %r" %_mk[0])
        print("fpi: %r" %_fpi[0])
        # pik_I32_chipt_nlo only takes two parameter values
        _res = pik_I32_chipt_nlo(_mpi, _mk, _fpi, p[0:3])
        if p.shape[0] > 2 and x.shape[1] > 3:
            # For the time being choose mpi**2 as lattice artefact
            _res += p[2]*x[:,3]**2
    return _res

def pik_I32_lo(p,x):
    """Wrapper for fitting the chipt formula of mu_pik_a_32 to the lattice data
    
    The chipt function for mu_pik a_32 gets evaluated and augmented with a
    possible lattice artefact. 

    Inputs
    ------
    p: 2d-array, array of shape (nboot,npar), the fit parameters, if npar > 2,
       last entry is trated as lattice artefact
    x: nd-array, measured input from lattice data (nboot,ninputs)

    Returns
    -------
    _res: 1d-array, samples of the variable mu_pik a_32
    """
    # The argument order for pik_I32_chipt_nlo is:
    # mpi, mk, fpi, p, lambda_x, meta
    _res = -x**2/(4.*np.pi)
    return _res

