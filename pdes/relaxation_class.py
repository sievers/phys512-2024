import numpy as np
from matplotlib import pyplot as plt
plt.ion()


def apply_bcs(V,x0,y0,y1,length):
    #set potential at edges to be zero
    V[:,0]=0
    V[:,-1]=0
    V[0,:]=0
    V[-1,:]=0

    #do our parallel plate

    ll=length//2
    V[x0-ll:x0+ll,y0]=1
    V[x0-ll:x0+ll,y1]=-1
    


n=64
V=np.zeros([n,n])
x0=n//2
y0=x0-n//20
y1=x0+n//20
length=n//8
apply_bcs(V,x0,y0,y1,length)


for i in range(100*n):
    V_new=(np.roll(V,1,0)+np.roll(V,-1,0)+np.roll(V,1,1)+np.roll(V,-1,1))/4
    rho=V-V_new
    V=V_new
    apply_bcs(V,x0,y0,y1,length)
    if (i%50==0):
        plt.figure(1)
        plt.clf()
        plt.imshow(V)
        plt.pause(0.0001)
        plt.figure(2)
        plt.clf()
        plt.imshow(np.log10(1e-12+np.abs(rho)))
        plt.colorbar()
        plt.pause(0.001)
