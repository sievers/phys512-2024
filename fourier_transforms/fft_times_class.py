import numpy as np
from scipy.fft import fft,rfft
import time

n1=2**20
n2=2**20-1
n3=2**20+7
n=[n1,n2,n3]
nn=len(n)
niter=10
for i in range(nn):
    v=np.random.randn(n[i])
    for iter in range(niter):
        t1=time.time()
        vft=fft(v)
        t2=time.time()
        print('with length ',n[i],'took ',t2-t1)
