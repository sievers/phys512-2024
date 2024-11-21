import numpy as np
from matplotlib import pyplot as plt
plt.ion()

n=3000

mat=np.random.randn(n,n)
mat=mat+mat.T
e,v=np.linalg.eigh(mat)
A=v@np.diag(np.abs(e)+0.01)@v.T
b=np.random.randn(n)

x=0*b
r=b-A@x
x_true=np.linalg.inv(A)@b
p=r.copy()
rtr=r@r

for i in range(10*n):
    Ap=A@p #new expensive step
    pAp=p@Ap
    alpha=rtr/pAp
    x=x+alpha*p
    r=r-alpha*Ap

    rtr_new=r@r
    beta=rtr_new/rtr
    p=r+beta*p
    
    rtr=rtr_new
    
    print('on iter ',i,' rtr is ',rtr)
    plt.clf()
    plt.plot(x_true,x,'.')
    plt.pause(0.001)
    if rtr<1e-8:
        break
