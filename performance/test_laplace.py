import numpy as np
import cupy as cp
import ctypes
import time
from cupyx.profiler import benchmark

mylib=ctypes.cdll.LoadLibrary("liblaplace.so")
apply_stencil=mylib.apply_stencil
apply_stencil.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_long,ctypes.c_long]

def swap(a,b):
    return b,a

def kernel_roll(V):
    return V-0.25*(cp.roll(V,1,0)+cp.roll(V,-1,0)+cp.roll(V,1,1)+cp.roll(V,-1,1))



n=4096
V=cp.zeros([n,n],dtype='float32')
V2=cp.zeros([n,n],dtype='float32')
x0=n//2
V[x0,x0]=1
apply_stencil(V2.data.ptr,V.data.ptr,n,n)

print('starting addresses: ',V.data.ptr,V2.data.ptr)
niter=10#*n
t1=time.time()
for i in range(niter):
    apply_stencil(V2.data.ptr,V.data.ptr,n,n)
    V,V2=swap(V,V2)
    V[x0,x0]=1
print(V[x0,x0+1])
t2=time.time()
print(V[x0,x0+2])
print('final addresses: ',V.data.ptr,V2.data.ptr)
print('time per iter: ',(t2-t1)/niter)
#rho=V-(cp.roll(V,1,0)+cp.roll(V,-1,0)+cp.roll(V,1,1)+cp.roll(V,-1,1))/4
rho=kernel_roll(V)
print('residual charge: ',cp.sum(cp.abs(rho[1:-1,1:-1]))-rho[x0,x0])
stats=benchmark(kernel_roll,(V,),n_repeat=100)
print(stats)
