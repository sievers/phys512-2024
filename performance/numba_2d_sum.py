import numpy as np
import numba as nb
import time



@nb.njit(parallel=True)
def sum_arr_rows(arr):
    tot=0
    n1=arr.shape[0]
    n2=arr.shape[1]
    for i in nb.prange(n1):
        for j in np.arange(n2):
            tot=tot+arr[i,j]
    return tot

@nb.njit(parallel=True)
def sum_arr_cols(arr):
    tot=0
    n1=arr.shape[0]
    n2=arr.shape[1]
    for j in nb.prange(n2):
        for i in np.arange(n1):
            tot=tot+arr[i,j]
    return tot



n=10000

mat=np.random.randn(n,n)
print('numpy sum is ',np.sum(mat))


niter=100
for i in range(niter):
    t1=time.time()
    #tot=sum_arr_rows(mat)
    tot=sum_arr_cols(mat)
    t2=time.time()
    print('took ',t2-t1,' with answer ',tot)
