import numpy as np
import scipy
import time
import numba as nb
from scipy.signal import convolve2d
import ctypes


mylib=ctypes.cdll.LoadLibrary("liblaplace_kernel.so")
apply_kernel_c=mylib.apply_kernel
apply_kernel_c.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int]




def kernel_ctypes(x):
    y=np.empty(x.shape)
    n=x.shape[0]
    m=x.shape[1]
    apply_kernel_c(x.ctypes.data,y.ctypes.data,n,m)
    return y

@nb.njit(parallel=False)
def kernel_numba(x):
    n=x.shape[0]
    m=x.shape[1]
    y=np.empty((n,m),np.float64)
    for i in nb.prange(1,n-1):
        for j in range(1,m-1):
            y[i,j]=x[i-1,j]+x[i+1,j]+x[i,j-1]+x[i,j+1]-4*x[i,j]
    return y

def kernel_python(x):
    n=x.shape[0]
    m=x.shape[1]
    y=np.empty((n,m),np.float64)
    for i in range(1,n-1):
        for j in range(1,m-1):
            y[i,j]=x[i-1,j]+x[i+1,j]+x[i,j-1]+x[i,j+1]-4*x[i,j]
    return y

def kernel_fft(x,kernelft):
    xft=np.fft.rfft2(x)
    return np.fft.irfft2(xft*kernelft)

def kernel_fft_scipy(x,kernelft):
    xft=scipy.fft.rfft2(x,workers=8)
    return scipy.fft.irfft2(xft*kernelft,workers=8)


n=4096
x=np.random.randn(n,n)
nrep=20

t1=time.time()
y1=np.roll(x,1,axis=0)+np.roll(x,-1,axis=0)+np.roll(x,1,axis=1)+np.roll(x,-1,axis=1)-4*x
t2=time.time()
print('did roll in ',t2-t1)

y2=kernel_numba(x) #run first so it compiles
t1=time.time()
for i in range(nrep):
    y2=kernel_numba(x)
t2=time.time()
print('did numba in ',(t2-t1)/nrep,np.std((y2-y1)[1:-1,1:-1]))

t1=time.time()
for i in range(nrep):
    y2=kernel_ctypes(x)
t2=time.time()
print('did ctypes in ',(t2-t1)/nrep,np.std((y2-y1)[1:-1,1:-1]))
    
kernel=np.zeros(x.shape)
kernel[0,0]=-4
kernel[0,1]=1
kernel[1,0]=1
kernel[0,-1]=1
kernel[-1,0]=1
kernelft=np.fft.rfft2(kernel)

t1=time.time()
y2=kernel_fft(x,kernelft)
t2=time.time()
print('did numpy kernel in ',t2-t1,np.std((y2-y1)[1:-1,1:-1]))
t1=time.time()
y2=kernel_fft_scipy(x,kernelft)
t2=time.time()
print('did scipy kernel in ',t2-t1,np.std((y2-y1)[1:-1,1:-1]))

tmp=np.zeros([3,3])
tmp[1,2]=1
tmp[1,0]=1
tmp[0,1]=1
tmp[2,1]=1
tmp[1,1]=-4
t1=time.time();
y2=convolve2d(x,tmp,'same');
t2=time.time();
print('did scipy convolve kernel in ',t2-t1,np.std((y2-y1)[1:-1,1:-1]))

if False:
    t1=time.time()
    y2=kernel_python(x)
    t2=time.time()
    print('did python loops in ',t2-t1,np.std((y2-y1)[1:-1,1:-1]))
