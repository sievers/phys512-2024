import numpy as np
from matplotlib import pyplot as plt
import numba as nb
import time
from scipy import fft
plt.ion()

@nb.njit
def grid_data(x,grid):
    ndata=x.shape[0]
    for i in np.arange(ndata):
        a=int(x[i,0])
        b=int(x[i,1])
        grid[a,b]=grid[a,b]+1
def cut_ob(x,npix):
    mask=x[:,0]>0
    mask=mask&(x[:,0]<npix)
    mask=mask&(x[:,1]>0)
    mask=mask&(x[:,1]<npix)
    return x[mask,:],mask
def inbounds(x,npix):
    x[:,0]=x[:,0]%npix
    x[:,1]=x[:,1]%npix
    return x
@nb.njit(parallel=True)
def get_forces(x,f,pot):
    ndata=x.shape[0]
    npix=pot.shape[0] #we're going to be lazy and assume square
    
    #bilinear: f(alpha,0)=f(0,0)*(1-alpha)+f(1,0)*alpha
    #          f(alpha,1)=f(0,1)*(1-alpha)+f(1,1)*alpha
    #f(alpha,beta)=(1-beta)*f(alpha,0)+beta*f(alpha,1)
    #=(1-beta)(1-alpha)f(0,0)+(1-beta)alpha f(0,1)
    #+beta(1-alpha)f(1,0)+beta alpha f(1,1)
    #grad:df/dalpha = (beta-1)f(0,0)+(1-beta)f(0,1)- beta f(1,0)+beta f(1,1)
    #     df/dbeta = (alpha-1)f(0,0)-alpha f(0,1)+(1-alpha)f(1,0)+alpha f(1,1)
    
    for i in nb.prange(ndata):
        a=int(x[i,0])
        b=int(x[i,1])
        alpha=x[i,0]-a
        beta=x[i,1]-b
        ar=a+1
        if ar>=npix:
            ar=ar%npix
        br=b+1
        if br>=npix:
            br=br%npix
        f[i,1]=((beta-1)*pot[a,b]+(1-beta)*pot[a,br]-beta*pot[ar,b]+beta*pot[ar,br])
        f[i,0]=((alpha-1)*pot[a,b]-alpha*pot[a,br]+(1-alpha)*pot[ar,b]+alpha*pot[ar,br])
        
        
def get_kernel(npix,soft=3,circ=True):
    kernel=np.zeros([npix,npix])

    if circ:
        x=np.fft.fftfreq(npix)*npix
    else:
        x=np.fft.fftfreq(2*npix)*2*npix
    xx,yy=np.meshgrid(x,x)
    rsqr=xx**2+yy**2
    rsqr[rsqr<soft**2]=soft**2
    kernel=1/np.sqrt(rsqr)
    if circ==False:
        kernel[npix//2:(3*npix//2),:]=0
        kernel[:,npix//2:(3*npix//2)]=0
    kernel=kernel/kernel.sum()
    kft=np.fft.rfft2(kernel)
    return kft

def get_pot(x,grid,kft):
    grid[:]=0
    grid_data(x,grid)
    gft=fft.rfft2(grid,workers=8)
    return fft.irfft2(gft*kft,workers=8)

def ics_2blob(n,npix):
    x=np.random.randn(n,2)*(npix/15)+npix/2
    x[::2,0]=x[::2,0]+npix/8
    x[1::2,0]=x[1::2,0]-npix/8
    return x
n=5000000
        
#x=np.zeros([n,2])
npix=2000
if True:
    #x=np.random.randn(n,2)*(npix/10)+npix/2
    x=np.random.rand(n,2)*npix
    v=0.0*x
else:
    x=ics_2blob(n,npix);
    x=inbounds(x,npix)
    v=0.0*x
    v[::2,1]=0.5
    v[1::2,1]=-0.5
grid=np.zeros([npix,npix])
grid_data(x,grid)
grid[:]=0
t1=time.time()
grid_data(x,grid)
t2=time.time()
print('elapsed time: ',t2-t1)
kft=get_kernel(npix)

pot=get_pot(x,grid,kft)
f=0*x
get_forces(x,f,pot)

dt=2.0
for i in range(10000):
    t1=time.time()
    x=x+v*dt
    inbounds(x,npix)
    grid[:]=0
    grid_data(x,grid)
    ta=time.time()
    pot=get_pot(x,grid,kft)
    tb=time.time()
    get_forces(x-0.5,f,pot)
    v=v+f*dt
    t2=time.time()
    print('elapsed time: ',t2-t1,tb-ta)
    plt.clf()
    plt.imshow(np.sqrt(grid),vmax=10)
    plt.pause(0.001)

