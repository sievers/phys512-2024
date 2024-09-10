import numpy as np
from matplotlib import pyplot as plt

def lorentz(x):
    return 1/(1+x**2)
def legendre_coeffs(n):
    x=np.linspace(-1,1,n)
    P=np.zeros([n,n])
    P[:,0]=1
    P[:,1]=x
    for i in range(1,n-1):
        P[:,i+1]=((2*i+1)*x*P[:,i]-i*P[:,i-1])/(i+1)
    Pinv=np.linalg.inv(P)
    wts=Pinv[0,:]
    
    return wts*(n-1)

def legendre_int(fun,x0,x1,n_targ,order):
    n_interval=int(np.round(n_targ/order))
    n=order*n_interval+1
    print(n)
    x=np.linspace(x0,x1,n)
    y=fun(x)
    coeffs=legendre_coeffs(order+1)
    dx=x[1]-x[0]
    tot=0
    for i in range(n_interval):
        i0=i*order
        tot=tot+np.sum(coeffs*y[i0:i0+order+1])
    return tot*dx

n=20
ord=6
if False:
    ans=legendre_int(np.exp,-1,1,n,ord)
    truth=np.exp(1)-np.exp(-1)
else:
    ans=legendre_int(lorentz,-1,1,n,ord)
    truth=np.arctan(1)-np.arctan(-1)
print('I got ',ans,' expected ',truth)
print('error: ',ans-truth,' for ',n,' points with order ',ord)
assert(1==0)
    
#P=legendre_coeffs(100)
Psimp=legendre_coeffs(3)
print(Psimp)
P5=legendre_coeffs(5)
print(P5,P5.sum())
#plt.ion()
#plt.clf()
#plt.plot(P[:,:5])
plt.show()
