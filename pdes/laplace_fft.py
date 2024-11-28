import numpy as np
from matplotlib import pyplot as plt
import time
def greens(n,ndim=3):
    #get the potential from a point charge at (0,0)
    dx=np.arange(n)
    dx[n//2:]=dx[n//2:]-n
    if ndim==2:
        pot=np.zeros([n,n])
        xmat,ymat=np.meshgrid(dx,dx)
        dr=np.sqrt(xmat**2+ymat**2)
        dr[0,0]=1 #dial something in so we don't get errors
        pot=np.log(dr)/2/np.pi
        pot=pot-pot[n//2,n//2]  #set it so the potential at the edge goes to zero
        pot[0,0]=pot[0,1]-0.25 #we know the Laplacian in 2D picks up rho/4 at the zero point
        return pot

def rho2pot(rho,kernelft):
    tmp=rho.copy()
    tmp=np.pad(tmp,(0,tmp.shape[0]))

    tmpft=np.fft.rfftn(tmp)
    tmp=np.fft.irfftn(tmpft*kernelft)
    if len(rho.shape)==2:
        tmp=tmp[:rho.shape[0],:rho.shape[1]]
        return tmp
    if len(rho.shape)==3:
        tmp=tmp[:rho.shape[0],:rho.shape[1],:rho.shape[2]]
        return tmp
    print("error in rho2pot - unexpected number of dimensions")
    assert(1==0)

def rho2pot_masked(rho,mask,kernelft,return_mat=False):
    rhomat=np.zeros(mask.shape)
    rhomat[mask]=rho
    potmat=rho2pot(rhomat,kernelft)
    if return_mat:
        return potmat
    else:
        return potmat[mask]


def cg(rhs,x0,mask,kernelft,niter,fun=rho2pot_masked,show_steps=False,step_pause=0.01):
    """cg(rhs,x0,mask,niter) - this runs a conjugate gradient solver to solve Ax=b where A
    is the Laplacian operator interpreted as a matrix, and b is the contribution from the 
    boundary conditions.  Incidentally, one could add charge into the region by adding it
    to b (the right-hand side or rhs variable)"""

    t1=time.time()
    Ax=fun(x0,mask,kernelft)
    r=rhs-Ax
    #print('sum here is ',np.sum(np.abs(r[mask])))
    p=r.copy()
    x=x0.copy()
    rsqr=np.sum(r*r)
    print('starting rsqr is ',rsqr)
    for k in range(niter):
        #Ap=ax_2d(p,mask)
        Ap=fun(p,mask,kernelft)
        alpha=np.sum(r*r)/np.sum(Ap*p)
        x=x+alpha*p
        if show_steps:            
            tmp=fun(x,mask,kernelft,True)
            plt.clf();
            plt.imshow(tmp,vmin=-2.1,vmax=2.1)
            plt.colorbar()
            plt.title('rsqr='+repr(rsqr)+' on iter '+repr(k+1))
            plt.savefig('laplace_iter_1024_'+repr(k+1)+'.png')
            plt.pause(step_pause)
        r=r-alpha*Ap
        rsqr_new=np.sum(r*r)
        beta=rsqr_new/rsqr
        p=r+beta*p
        rsqr=rsqr_new
        #print('rsqr on iter ',k,' is ',rsqr,np.sum(np.abs(r[mask])))
    t2=time.time()
    print('final rsqr is ',rsqr,' after ',t2-t1,' seconds')
    return x




def get_rhs(pot):
    tot=0
    ndim=len(pot.shape)
    for dim in range(ndim):
        tot=tot+np.roll(pot,1,dim)
        tot=tot+np.roll(pot,-1,dim)
    return tot
def get_rho(pot):
    tmp=get_rhs(pot)
    return 4*pot-tmp
plt.ion()


n=1024
bc=np.zeros([n,n])
mask=np.zeros([n,n],dtype='bool')
mask[0,:]=True
mask[-1,:]=True
mask[:,0]=True
mask[:,-1]=True
bc[0,:]=0.0
bc[0,0]=0.0
bc[0,-1]=0.0
#This adds a bar in the interior held at fixed potential
bc[n//4:3*n//4,(2*n//5)]=2.0
mask[n//4:3*n//4,(2*n//5)]=True

bc[n//4:3*n//4,(3*n//5)]=-2.0
mask[n//4:3*n//4,(3*n//5)]=True

kernel=greens(2*n,2)
kernelft=np.fft.rfft2(kernel)
#fwee=rho2pot(bc,kernelft)
#rr=bc[mask]
#fwee2=rho2pot_masked(rr,mask,kernelft)


rhs=bc[mask]
x0=0*rhs

rho_out=cg(rhs,x0,mask,kernelft,40,show_steps=True,step_pause=0.25)
pot=rho2pot_masked(rho_out,mask,kernelft,True)


