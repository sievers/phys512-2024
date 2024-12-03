import numpy as np
import cupy as cp
import ctypes
import time
from cupyx.profiler import benchmark

mylib=ctypes.cdll.LoadLibrary("libaddVecs.so")
add_cuda_c=mylib.add2
add_cuda_c.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_long]
add_cuda_sizes=mylib.add3
add_cuda_sizes.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_long,ctypes.c_long,ctypes.c_long]

def add_cuda(c,a,b,nblock=64,bs=256):
    n=len(a)
    #add_cuda_sizes(c.data.ptr,a.data.ptr,b.data.ptr,n,nblock,bs)
    #print('calling add_cuda with n ',n)    
    add_cuda_sizes(c.data.ptr,a.data.ptr,b.data.ptr,n,nblock,bs)
    #add_cuda_c(c.data.ptr,a.data.ptr,b.data.ptr,n)


n=2048*2048*16
a=cp.ones(n,dtype='float32')
b=cp.ones(n,dtype='float32')
c=cp.zeros(n,dtype='float32')
print(c[0])
t1=time.time()
#add_cuda(c.data.ptr,a.data.ptr,b.data.ptr,n)
add_cuda(c,a,b)
print(c[0])
t2=time.time()
print('time to add was ',t2-t1)
print('c mean/std are ',cp.mean(c),cp.std(c))
print('effective memory bandwidth: ',n*3*4/(t2-t1)/1024**3,' GB/s')


if True:
    nblock=192
    bs=256
    stats=benchmark(add_cuda,(c,a,b,nblock,bs),n_repeat=100)
    print(stats)
    t_ave=np.mean(stats.gpu_times)
    print('effective memory bandwidth in benchmark: ',n*3*4/t_ave/1024**3,' GB/s')


aa=cp.asnumpy(a)
bb=cp.asnumpy(b)
cc=0*aa
cc[:]=aa[:]+bb[:]
t1=time.time()
cc[:]=aa[:]+bb[:]
t2=time.time()
print('time on cpu is ',t2-t1,' with bandwith ',3*4*n/(t2-t1)/1024**3)
