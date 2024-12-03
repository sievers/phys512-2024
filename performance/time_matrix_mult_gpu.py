import numpy as np
import cupy as cp
import time

n=[100,300,1000,3000,5000,7000,10000]
nrep=[1000,200,20,3,3,1,1,1]

for i,nn in enumerate(n):
    #x=cp.random.rand(nn,nn,dtype='float32')
    x=cp.random.rand(nn,nn) #changed one character!
    t1=time.time()
    for j in range(nrep[i]):
        y=x@x
    #print(y[0,0])
    t2=time.time()
    nop=2*nn**3*nrep[i]
    gflops=nop/(t2-t1)/1e9
    print('got ',gflops,' gigaflops on matrix size of ',nn)
