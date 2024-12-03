import numpy as np
import time
from cupyx.profiler import benchmark
import cupy as cp

n=[100,300,1000,3000,5000,7000,10000,15000]
nrep=[1000,200,20,3,3,1,1,1]
tmp=0.0
for i,nn in enumerate(n):
    #x=cp.random.rand(nn,nn,dtype='float32')
    x=cp.random.rand(nn,nn) #changed one character!
    t1=time.time()
    for j in range(nrep[i]):
        y=x@x
    #print(y[0,0])    #this line makes a huge difference in reported timing
    tmp=tmp+cp.asnumpy(y[0,0]) #this line also syncs the streams
    t2=time.time()
    #print(tmp)
    nop=2*nn**3*nrep[i]
    gflops=nop/(t2-t1)/1e9
    print('got ',gflops,' gigaflops on matrix size of ',nn, ' with time ',(t2-t1)/nrep[i])
    #print(benchmark(cp.matmul,(x,x),n_repeat=nrep[i]))
