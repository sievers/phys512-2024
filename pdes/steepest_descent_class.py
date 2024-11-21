import numpy as np
from matplotlib import pyplot as plt
plt.ion()

n=1000

mat=np.random.randn(n,n)
mat=mat+mat.T
e,v=np.linalg.eigh(mat)
A=v@np.diag(np.abs(e)+0.01)@v.T
b=np.random.randn(n)

x=0*b
r=b-A@x
x_true=np.linalg.inv(A)@b
for i in range(10*n):
    Ar=A@r #this is the expensive step
    rtr=r@r
    rAr=r@Ar
    alpha=rtr/rAr
    x=x+alpha*r
    r=r-alpha*Ar
    print('rtr is ',rtr)
    plt.clf()
    plt.plot(x_true,x,'.')
    plt.pause(0.001)
