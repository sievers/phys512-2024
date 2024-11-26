import numpy as np
import numba as nb
import time
@nb.njit(parallel=True)
def fill(b,a):
    n=a.shape[0]
    m=a.shape[1]
    for i in nb.prange(n):
        for j in np.arange(m):
            b[i,j]=a[i,j]


n=10000
a=np.random.randn(n,n)
b=0*a

for i in range(10):
    t1=time.time()
    fill(b,a)
    t2=time.time()
    print('numba: ',t2-t1,(2*b.nbytes)/(t2-t1)/1e9)
for i in range(10):
    t1=time.time()
    b[:]=a[:]
    t2=time.time()
    print('numpy fill: ',t2-t1,(2*b.nbytes)/(t2-t1)/1e9)

for i in range(10):
    t1=time.time()
    b=a.copy()
    t2=time.time()
    print('numpy copy: ',t2-t1,(2*b.nbytes)/(t2-t1)/1e9)
