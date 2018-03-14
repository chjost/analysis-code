from scipy.optimize import fsolve
import numpy as np

def corr_arcosh(data,T,add=None):
    mass = np.zeros_like(data[:,:-2])
    for b, row in enumerate(data):
        for t in range(1, len(row)-1):
            mass[b,t-1] = (row[t-1] + row[t+1])/(2.*row[t])
    return np.arccosh(mass)

def corr_exp_implement(m,r0,r1,t,T):
    """
    Parameters
    ----------
    p: tuple
    """
    _den = np.exp(-m*t) + np.exp(-m*(T-t))
    _num = np.exp(-m*(t+1)) + np.exp(-m*(T-(t+1))) 
    _diff = r0/r1 - _den/_num 
    return _diff

def corr_exp(data,T,add=None): 
    mass = np.zeros_like(data[:,:-1])
    print("Exponential solve for symmetric correlator")
    print(mass.shape)
    for b, row in enumerate(data):
        for t in range(len(row)-1):
             mass[b, t] = fsolve(corr_exp_implement,0.5,args=(row[t],row[t+1],t,T))
    return mass

def corr_exp_asym_implement(m,r0,r1,t,T):
    """
    Parameters
    ----------
    p: tuple
    """
    _den = np.exp(-m*t) - np.exp(-m*(T-t))
    _num = np.exp(-m*(t+1)) - np.exp(-m*(T-(t+1))) 
    _diff = r0/r1 - _den/_num 
    return _diff

def corr_exp_asym(data,T,add=None): 
    mass = np.zeros_like(data[:,:-1])
    print("Exponential solve for symmetric correlator")
    print(mass.shape)
    for b, row in enumerate(data):
        for t in range(len(row)-1):
             mass[b, t] = fsolve(corr_exp_asym_implement,0.5,args=(row[t],row[t+1],t,T))
    return mass

def corr_log(data,T=None,add=None):
    mass = np.zeros_like(data[:,:-1])
    for b, row in enumerate(data):
        for t in range(len(row)-2):
           mass[b, t] = np.log(row[t]/row[t+1])
    return mass

def corr_shift_weight_implement(m,r0,r1,t,T,weight,shift=1.):
    _num = np.exp(-m*t) + np.exp(-m*(T-t))-np.exp(weight*shift) * ( np.exp(-m*(t+shift)) + np.exp(-m*(T-t-shift)) )
    _den = np.exp(-m*(t+1)) + np.exp(-m*(T-t-1))-np.exp(weight*shift) * ( np.exp(-m*(t+1+shift)) + np.exp(-m*(T-t-1-shift)) )
    _diff = r0/r1 - _num/_den
    return _diff

def corr_shift_weight(data,T,add):
    mass = np.zeros_like(data[:,:-1])
    print(mass.shape)
    for b, row in enumerate(data):
         for t in range(len(row)-1):
              mass[b, t] = fsolve(corr_shift_weight_implement,0.5,
                                  args=(row[t],row[t+1],t,T,add[0][b]),
                                  maxfev=1000)
    return mass 

def pik_pollution(t,T,epi,ek):
    pollution = np.exp(-epi*t) * np.exp(-ek*(T-t)) + np.exp(-ek*t) * np.exp(-epi*(T-t))
    return pollution

def pik_div(epik,row,row_shifted,t,T,epi,ek):
    c_t = np.exp(-epik*t)+np.exp(-epik*(T-t)) 
    c_t1 = np.exp(-epik*(t+1))+np.exp(-epik*(T-(t+1)))
    c_t2 = np.exp(-epik*(t+2))+np.exp(-epik*(T-(t+2))) 
    p_t = pik_pollution(t,T,epi,ek)
    p_t1 = pik_pollution(t+1,T,epi,ek)
    p_t2 = pik_pollution(t+2,T,epi,ek) 
    num = c_t - p_t/p_t1 * c_t1
    den = c_t1 - p_t1/p_t2 * c_t2
    #num = np.exp(-epik*t)+np.exp(-epik*(T-t)) - pik_pollution(t,T,epi,ek)/pik_pollution(t+1,T,epi,ek) * (np.exp(-epik*(t+1))+np.exp(-epik*(T-(t+1))))
    #den = np.exp(-epik*(t+1))+np.exp(-epik*(T-(t+1))) - pik_pollution(t+1,T,epi,ek)/pik_pollution(t+2,T,epi,ek) * (np.exp(-epik*(t+2))+np.exp(-epik*(T-(t+2))))
    return row/row_shifted - num/den

def corr_shift_weight_div(data,T,add):
    # add[0] is pion energy
    # add[1] is kaon energy
    # add is a list of bootstrapsamples: [epi,ek]
    print("T for effective mass is %d" %T)
    print(add[0][0],add[1][0])
    mass = np.zeros_like(data[:,:-2])
    print("Exponential solve for symmetric correlator")
    print(mass.shape)
    for b, row in enumerate(data):
        for t in range(len(row)-2):
             mass[b, t] = fsolve(pik_div,0.5,
                                 args=(row[t],row[t+1],t,T,add[0][b],add[1][b]))
    return mass

