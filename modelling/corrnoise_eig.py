import numpy as np
from scipy.linalg import toeplitz
from matplotlib import pyplot as plt
plt.ion()

x=np.arange(1000)
mycorr=np.exp(-0.5*x**2/100**2)

N=toeplitz(mycorr)
N_slow=np.zeros_like(N)
dcorr=np.zeros(2*len(mycorr)-1)
dcorr[:len(mycorr)-1]=np.flipud(mycorr[1:])
dcorr[len(mycorr)-1:]=mycorr
n=len(x)
for i in range(n):
    N_slow[i,:]=dcorr[i,n-i:(2*n-i)]


plt.figure(1)
plt.clf()
plt.imshow(N)
plt.show()

e,v=np.linalg.eigh(N)
#e[:-2]=0
e[e<0]=0 #make sure we have no negative variance
mysig=np.sqrt(e)

mydat_uncorr=mysig*np.random.randn(len(mysig))

mydat=v@mydat_uncorr
plt.figure(2)
plt.clf()
plt.plot(mydat)
plt.show()
