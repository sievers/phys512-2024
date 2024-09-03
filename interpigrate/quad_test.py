import numpy as np
from scipy.integrate import quad

def gfun(x,sig=0.01,x0=0):
    return 1+np.exp(-0.5*(x-x0)**2/(2*sig**2))/sig


x0=-20+2
x1=20+2
print('area from ',x0,' to ',x1,' is ',quad(gfun,x0,x1))
