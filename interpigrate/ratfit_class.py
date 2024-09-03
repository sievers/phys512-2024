import numpy as np
from matplotlib import pyplot as plt
plt.ion()

def fit_rat(x,y,nump,nq):
    n=len(x)
    assert(nump+nq+1==n)
    A=np.zeros([n,n])
    for i in range(nump+1):
        A[:,i]=x**i
    for i in range(nq):
        A[:,nump+1+i]=-y*x**(i+1)
    fitp=np.linalg.inv(A)@y
    return fitp

x=np.linspace(-1,1,7)
y=np.exp(-0.5*x**2)
fit=fit_rat(x,y,3,3)

