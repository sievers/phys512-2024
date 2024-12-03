import numpy as np;print("using numpy")
#import cupy as np;print("using cupy")
from cupyx.profiler import benchmark

def kernel(V):
    return V-0.25*(np.roll(V,1,0)+np.roll(V,-1,0)+np.roll(V,1,1)+np.roll(V,-1,1))

n=2048
V=np.zeros([n,n],dtype='float32')
x0=n//2
V[x0,x0]=1
stats=benchmark(kernel,(V,),n_repeat=100)
print("cpu/gpu times (msec): ",np.mean(stats.cpu_times)*1000,np.mean(stats.gpu_times)*1000)
#print(stats)
