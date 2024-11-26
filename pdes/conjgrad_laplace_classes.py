import numpy as np
from matplotlib import pyplot as plt
plt.ion()

class MatAsVec:
    def __init__(self,mat):
        self.mat=mat.copy()
    def __matmul__(self,other):
        return np.sum(self.mat*other.mat)
    def __mul__(self,scalar):
        return MatAsVec(self.mat*scalar)
    def __rmul__(self,scalar):
        return self.__mul__(scalar)
    def __add__(self,other):
        return MatAsVec(self.mat+other.mat)
    def __sub__(self,other):
        return MatAsVec(self.mat-other.mat)
    def copy(self):
        return MatAsVec(self.mat.copy())
def bcs_plates(n,pars):
    
    dx=pars[0]
    dy=pars[1]
    V0=pars[2]

    x=np.linspace(-1,1,n)
    mask=np.zeros([n,n],dtype='bool')
    bcs=np.zeros([n,n])

    mask[:]=False
    bcs[:,0]=0
    mask[:,0]=True
    bcs[:,-1]=0
    mask[:,-1]=True
    bcs[0,:]=0
    mask[0,:]=True
    bcs[-1,:]=0
    mask[-1,:]=True

    x0_targ=dx/2
    x0=np.argmin(np.abs(x-x0_targ))
    x1=np.argmin(np.abs(x+x0_targ))

    y0_targ=dy/2
    y0=np.argmin(np.abs(x+y0_targ))
    y1=np.argmin(np.abs(x-y0_targ))+1
    print('x0,x1,y0,y1 are ',x0,x1,y0,y1)
    bcs[x0,y0:y1]=V0
    mask[x0,y0:y1]=True
    bcs[x1,y0:y1]=-V0
    mask[x1,y0:y1]=True
    return bcs,mask
class Laplace:
    #def __init__(self,n,bc_fun,pars):
    def __init__(self,bcs,mask):
        #self.n=n
        #self.bcs,self.mask=bc_fun(n,pars)
        self.bcs=bcs.copy()
        self.mask=mask.copy()
    def apply(self,mat):
        tmp=(np.roll(mat,1,0)+np.roll(mat,-1,0)+np.roll(mat,1,1)
             +np.roll(mat,-1,1))/4.0
        return tmp
    def __matmul__(self,vec):
        mm=vec.mat.copy()
        mm[self.mask]=0
        tmp=self.apply(mm)
        tmp[self.mask]=0
        return MatAsVec(mm-tmp)
    def get_rhs(self):
        #right-hand side is a laplace iter of the boundary conditions,
        #but read out at the non-boundary conditions, with a minus sign
        tmp=self.apply(self.bcs)
        tmp[self.mask]=0
        return MatAsVec(tmp)
        
def conjgrad(A,b,niter,x=None):
    if x is None:
        x=0*b
    Ax=A@x
    
    print(type(b))
    r=b-Ax
    print(type(r))

    p=r.copy()
    rtr=r@r
    print('rtr starting is ',rtr)
    for i in range(niter):
        Ap=A@p
        pAp=p@Ap
        alpha=rtr/pAp
        x=x+alpha*p
        r=r-alpha*Ap
        rtr_new=r@r
        beta=rtr_new/rtr
        p=r+beta*p
        rtr=rtr_new
        print("on iter ",iter," rtr is ",rtr)
    return x


n=100
A=np.random.randn(n,n)
A=A+A.T+n*np.eye(n)
b=np.random.randn(n)
#x=conjgrad(A,b,10)

print('now solving laplace')
pars=[0.2,0.4,1.0]
n=100
bcs,mask=bcs_plates(n,pars)
A=Laplace(bcs,mask)
b=A.get_rhs()
x=conjgrad(A,b,n*2)
V=x.mat+A.bcs
rho=V-A.apply(V)

y=0*b
for i in range(2000):
    y=A@y
    y.mat[A.mask]=A.bcs[A.mask]
crud=A.apply(y.mat)-y.mat
