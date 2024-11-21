#code to solve AX=b using steepest descent
#we're minimizing 1/2 x^T A x -x^T b
#which has gradient Ax-b = -r
import numpy as np
from matplotlib import pyplot as plt
plt.ion()

#first, make a positive-definite matrix
n=1000
mat=np.random.randn(n,n)
mat=mat+mat.T
e,v=np.linalg.eigh(mat)
A=v@np.diag(0.1+np.abs(e))@v.T
b=np.random.randn(n) #our RHS

#do the exact solution so we can see how we're doing
x_true=np.linalg.inv(A)@b

#we'll start with x=0 as our guess
x=0*b
r=b-A@x
for i in range(10*n):    
    Ar=A@r  #this is the only step in which we access A
    alpha=r@r/(r@Ar)
    x=x+alpha*r
    r=r-alpha*Ar
    print('iter ',i,' has rsqr ',r@r) #this should trend downwards
    plt.clf()
    plt.plot(x_true,x,'.')
    plt.pause(0.001)

plt.clf()
plt.plot(x_true,x,'.')
plt.show()
