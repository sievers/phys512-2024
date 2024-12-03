import numpy as np
import time

n=[100,300,1000,3000,5000,7000,10000,15000]
nrep=[1000,200,20,3,3,1,1,1]

for i,nn in enumerate(n):
    x=np.random.rand(nn,nn)
    t1=time.time()
    for j in range(nrep[i]):
        y=x@x
    t2=time.time()
    nop=2*nn**3*nrep[i]
    gflops=nop/(t2-t1)/1e9
    print('got ',gflops,' gigaflops on matrix size of ',nn)
