import numpy as np
from matplotlib import pyplot as plt

def cheb_mat_uniform(nx,ord):
    x=np.linspace(-1,1,nx)
    mat=np.zeros([nx,ord+1])
    mat[:,0]=1.0
    if ord>0:
        mat[:,1]=x
    if ord>1:
        for i in range(1,ord):
            mat[:,i+1]=2*x*mat[:,i]-mat[:,i-1]
    return mat,x

n=5000
ord=150
mat,x=cheb_mat_uniform(n,ord)
y=np.sin(x*np.pi)

#we'll do a least-squares fit, details coming Friday
lhs=np.dot(mat.transpose(),mat)
rhs=np.dot(mat.transpose(),y)
fitp=np.dot(np.linalg.inv(lhs),rhs)

ncoeff=5
mat=mat[:,1::2]
fitp=fitp[1::2]
pred=np.dot(mat[:,:ncoeff],fitp[:ncoeff])



lhs2=np.dot(mat[:,:ncoeff].transpose(),mat[:,:ncoeff])
rhs2=np.dot(mat[:,:ncoeff].transpose(),y)
fitp2=np.dot(np.linalg.inv(lhs2),rhs2)
pred2=np.dot(mat[:,:ncoeff],fitp2)

plt.clf();plt.plot(x,pred-y);plt.plot(x,pred2-y);
plt.legend(['Chebyshev Residual','Least-Squares Residual'])
plt.savefig('cheb_lss_resids.png')

print('rms error for least-squares is ',np.sqrt(np.mean((pred2-y)**2)),' with max error ',np.max(np.abs(pred2-y)))
print('rms error for chebyshev     is ',np.sqrt(np.mean((pred-y)**2)),' with max error ',np.max(np.abs(pred-y)))



    
