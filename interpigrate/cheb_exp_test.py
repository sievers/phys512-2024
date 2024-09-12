import numpy as np
from matplotlib import pyplot as plt
plt.ion()

def cheb_mat(x,ord):
    A=np.zeros([len(x),ord])
    A[:,0]=1
    if ord>0:
        A[:,1]=x
    if ord>1:
        for i in range(1,ord-1):
            A[:,i+1]=2*x*A[:,i]-A[:,i-1]
    return A

def cheb_eval(coeffs,x):
    A=cheb_mat(x,len(coeffs))
    return A@coeffs

def cheb_fit(fun,x):
    A=cheb_mat(x,len(x))
    y=fun(x)
    coeffs=np.linalg.inv(A)@y
    return coeffs
x=np.linspace(-1,1,31)
coeffs=cheb_fit(np.exp,x)

xx=np.linspace(-1,1,1001)

nuse=6
fitp=coeffs[:nuse]

A=cheb_mat(xx,nuse)
yy=A@fitp
y=np.exp(xx)

ypow=1
for i in range(1,nuse):
    ypow=ypow+xx**i/np.prod(np.arange(1,i+1))

cheb_err=y-yy
tayl_err=y-ypow
plt.clf()
plt.plot(xx,cheb_err)
plt.plot(xx,tayl_err)
plt.show()
