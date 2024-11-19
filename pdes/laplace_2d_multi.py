import numpy as np
from matplotlib import pyplot as plt
import time

def make_rhs_2d(pot,mask):
    """Make the right hand side.  We know that in charge-free regions,
    V0 equals the average of its neighbors, or V0+0.25*(V_l+V_r +V_u+V_d)=0.  If
    some of the neighbors have been set by the boundary conditions, we have
    V0 + 0.25*sum(V interior) = -0.25*sum(V_boundary) and we only solve for 
    the potential not on the specified boundary conditions.  Note that if we had charge in there, 
    then up to a multiplicative constant, V0+0.25*(V_l+V_r +V_u+V_d)=rho, so you could add the 
    (suitably scaled) charge to the rhs."""
    mat=np.zeros(pot.shape)
    mat[:,:-1]=mat[:,:-1]+pot[:,1:]
    mat[:,1:]=mat[:,1:]+pot[:,:-1]
    mat[:-1,:]=mat[:-1,:]+pot[1:,:]
    mat[1:,:]=mat[1:,:]+pot[:-1,:]
    mat[mask]=0  #for the masked region, i.e. specified by the boundary conditions, we'll peg the RHS to zero since we keep 
                 #the potential fixed anyways
    return mat

def ax_2d(mat,mask,copy=False):
    """Write the Laplacian operator in the way we need it to be.  Note that the boundary conditions as specified by the mask
    do not enter into the matrix since they are on the right-hand side of the matrix equation.  So, set them to zero here, then we won't have
    to worry about handling them separately."""
    if copy:
        mat=mat.copy()
    mat[mask]=0
    mm=4*mat
    mm[:,:-1]=mm[:,:-1]-mat[:,1:]
    mm[:,1:]=mm[:,1:]-mat[:,:-1]
    mm[1:,:]=mm[1:,:]-mat[:-1,:]
    mm[:-1,:]=mm[:-1,:]-mat[1:,:]
    mm[mask]=0
    return mm
    
def cg(rhs,x0,mask,niter):
    """cg(rhs,x0,mask,niter) - this runs a conjugate gradient solver to solve Ax=b where A
    is the Laplacian operator interpreted as a matrix, and b is the contribution from the 
    boundary conditions.  Incidentally, one could add charge into the region by adding it
    to b (the right-hand side or rhs variable)"""

    t1=time.time()
    Ax=ax_2d(x0,mask)
    r=rhs-Ax
    #print('sum here is ',np.sum(np.abs(r[mask])))
    p=r.copy()
    x=x0.copy()
    rsqr=np.sum(r*r)
    print('starting rsqr is ',rsqr)
    for k in range(niter):
        Ap=ax_2d(p,mask)
        alpha=np.sum(r*r)/np.sum(Ap*p)
        x=x+alpha*p
        r=r-alpha*Ap
        rsqr_new=np.sum(r*r)
        beta=rsqr_new/rsqr
        p=r+beta*p
        rsqr=rsqr_new
        #print('rsqr on iter ',k,' is ',rsqr,np.sum(np.abs(r[mask])))
    t2=time.time()
    print('final rsqr is ',rsqr,' after ',t2-t1,' seconds')
    return x

def deres_mat(mat):
    """A quick and dirty way to downgrade the resolution of a potential matrix by a 
    factor of 2.  Since this is taking the maximum, you should not trust this very much if you
    have specified negative potentials anywhere."""
    mm=np.zeros([mat.shape[0]//2,mat.shape[1]//2],dtype=mat.dtype)
    mm=np.maximum(mm,mat[::2,::2])
    mm=np.maximum(mm,mat[::2,1::2])
    mm=np.maximum(mm,mat[1::2,::2])
    mm=np.maximum(mm,mat[1::2,1::2])
    return mm
def upres_mat(mat):
    """A quick & dirty way to increase the resolutio of a potential matrix by a factor
    of 2.  A smarter version here would lead to faster convergence, but I have ignored that."""
    mm=np.zeros([mat.shape[0]*2,mat.shape[1]*2],dtype=mat.dtype)
    mm[::2,::2]=mat
    mm[::2,1::2]=mat
    mm[1::2,::2]=mat
    mm[1::2,1::2]=mat
    return mm



npix=1024
#set boundary conditions on the edges:  V=1 on the top/bottom, 0 on the left, right.  You could 
#choose otherwise.
bc=np.zeros([npix,npix])
bc[:,0]=0
bc[:,-1]=0
#bc[0,0]=0.5
#bc[-1,0]=0.5
#bc[-1,-1]=0.5
#bc[0,-1]=0.5

mask=np.zeros([npix,npix],dtype='bool')
mask[0,:]=1
mask[-1,:]=1
mask[:,0]=1
mask[:,-1]=1


#add a feature in the middle of the region, held at a constant potential.
#bc[npix//2,npix//4:3*npix//4]=2
#mask[npix//2,npix//4:3*npix//4]=1

#add a plate in the middle with a hole in it
bc[npix//2,:npix//3]=1
mask[npix//2,:npix//3]=True
bc[npix//2,-npix//3:]=1
mask[npix//2,-npix//3:]=True


npass=6
#loop through the resolutions.  In this case, start with something 2**6 times coarser than the desired resolution
#solve it, and then increase the resolution by a factor of 2.  Solve the next-higher resolution problem using that
#as the starting guess.
all_masks=[None]*npass
all_bc=[None]*npass
all_rhs=[None]*npass
all_x=[None]*npass
all_masks[0]=mask
all_bc[0]=bc
for i in range(1,npass):
    all_masks[i]=deres_mat(all_masks[i-1])
    all_bc[i]=deres_mat(all_bc[i-1])


#now, make the lowest-resolution map.  First set up the RHS given the low-res boundary conditions/masks we already made.
nn=all_masks[-1].shape[0]
niter=3*nn
all_rhs[-1]=make_rhs_2d(all_bc[-1],all_masks[-1])
all_x[-1]=cg(all_rhs[-1],0*all_rhs[-1],all_masks[-1],niter)

#show our lowest-resolution solution
plt.ion();
plt.imshow(all_x[-1])
plt.pause(0.01)
#and now, run a loop where you increase the resolution, solve for a fixed number of iterations, then repeat until 
#you're at your desired resolution.  The hardwired number of iterations is rather sloppy, but gets the job done.
niter=150

for i in range(npass-2,-1,-1):
    all_rhs[i]=make_rhs_2d(all_bc[i],all_masks[i])
    x0=upres_mat(all_x[i+1])
    all_x[i]=cg(all_rhs[i],x0,all_masks[i],niter)
    #plot the current-resolution potential
    plt.clf()
    plt.imshow(all_x[i])
    plt.pause(3)

#finally, paste in the boundary conditions since we didn't solve for them in conjugate gradient.
for i in range(npass):
    all_x[i][all_masks[i]]=all_bc[i][all_masks[i]]

plt.clf();plt.imshow(all_x[0])  #and finally, plot the final map with boundary conditions enforced.
plt.savefig('laplace_2d_solution.png')

plt.clf();plt.plot(all_x[0][npix//2,:])
plt.savefig('potential_in_middle.png')

#now work out our final charge distribution.
xx=all_x[0]
rho=4*xx[1:-1,1:-1];rho=rho-xx[2:,1:-1];rho=rho-xx[:-2,1:-1];rho=rho-xx[1:-1,2:];rho=rho-xx[1:-1,:-2]
plt.clf();plt.plot(rho[npix//2-1,:]);
plt.savefig('charge_in_middle.png')
                           
#assert(1==0)



#rhs=make_rhs_2d(bc,mask)
#xx=cg(rhs,0*rhs,mask,300)
#xx[mask]=bc[mask]

#xx2=cg(rhs,xx,mask,20)
#xx3=cg(rhs,xx2,mask,20)

