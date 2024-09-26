import numpy as np
from scipy.linalg import toeplitz
from matplotlib import pyplot as plt
plt.ion()

def gen_corrmat(N,nsamp,white=0.0001):
    x=np.arange(N)
    mycorr=np.exp(-0.5*(x**2/nsamp**2))
    mycorr[0]=mycorr[0]+white
    corrmat=toeplitz(mycorr)
    return corrmat

def gen_data(corrmat):
    e,v=np.linalg.eigh(corrmat)
    d=np.random.randn(len(e))
    d_scale=d*np.sqrt(e)
    d=v@(d_scale)
    return d

N=1000
C10=gen_corrmat(N,10)
d10=gen_data(C10)
plt.clf()
plt.plot(d10)
plt.show()

C100=gen_corrmat(N,100)
d100=gen_data(C100)
plt.plot(d100)

x=np.arange(N)
x0=500
sig=100
model=np.exp(-0.5*(x-x0)**2/sig**2)
plt.plot(model)

d100_2=d100+model #my 100-pixel correlated noise plus signal
plt.plot(d100_2)
Ninv_100=np.linalg.inv(C100)
lhs=model@Ninv_100@model
rhs=model@Ninv_100@d100_2
amp_100=rhs/lhs

e100=np.sqrt(1/lhs)
print('amplitude and error: ',amp_100,e100)


d10_2=d10+model #my 100-pixel correlated noise plus signal
plt.plot(d10_2)
Ninv_10=np.linalg.inv(C10)
lhs=model@Ninv_10@model
rhs=model@Ninv_10@d10_2
amp_10=rhs/lhs
e10=1/np.sqrt(lhs)
print('10 pix amp/err: ',amp_10,e10)
