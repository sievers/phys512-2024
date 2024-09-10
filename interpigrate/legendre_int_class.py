import numpy as np
from matplotlib import pyplot as plt
plt.ion()

def legmat(n):
    x=np.linspace(-1,1,n)
    A=np.zeros([n,n])
    A[:,0]=1
    if n>0:
        A[:,1]=x
    if n>1:
        for i in range(1,n-1):
            tot=(2*i+1)*x*A[:,i]-i*A[:,i-1]
            A[:,i+1]=tot/(i+1)
    return A


A=legmat(57)
Ainv=np.linalg.inv(A)
coeffs=Ainv[0,:]
print(coeffs)
