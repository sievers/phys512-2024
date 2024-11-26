import numpy as np
from matplotlib import pyplot as plt
plt.ion()

def apply_stencil(A,do_A=True):
    tot=np.roll(A,1,axis=0)+np.roll(A,-1,axis=0)+np.roll(A,1,axis=1)+np.roll(A,-1,axis=1)
    if do_A:
        return A-tot/4
    else:
        return -tot/4

def Ax(V,mask):
    #Vuse=np.zeros([V.shape[0]+2,V.shape[1]+2])
    #Vuse[1:-1,1:-1]=V
    Vuse=V.copy()
    Vuse[mask]=0
    ans=apply_stencil(Vuse)
    ans[mask]=0
    #ans=(Vuse[1:-1,:-2]+Vuse[1:-1,2:]+Vuse[2:,1:-1]+Vuse[:-2,1:-1])/4.0
    #ans=ans-V[1:-1,1:-1]

    return ans

def pad(A):
    AA=np.zeros([A.shape[0]+2,A.shape[1]+2])
    AA[1:-1,1:-1]=A
    return AA
n=100

V=np.zeros([n,n])
bc=0*V

mask=np.zeros([n,n],dtype='bool')
mask[:,0]=True
mask[:,-1]=True
mask[0,:]=True
mask[-1,:]=True
#mask[n//2,n//4:(3*n)//4]=True
mask[n//4:n//2,n//4:n//2]=True
bc[n//4:n//2,n//4:n//2]=1.0
mask[n//4:n//2,n//2:(3*n)//4]=True
bc[n//4:n//2,n//2:(3*n)//4]=-1.0

b=-apply_stencil(bc,False)
b[mask]=0
#b2=-(bc[1:-1,0:-2]+bc[1:-1,2:]+bc[:-2,1:-1]+bc[2:,1:-1])/4.0

V=0*bc



r=b-Ax(V,mask)
p=r.copy()

for k in range(n):
    #Ap=(Ax(pad(p),mask))
    Ap=Ax(p,mask)
    #rtr=np.sum(r*r)
    rtr=np.sum(p*r)
    print('on iteration ' + repr(k) + ' residual is ' + repr(rtr))
    alpha=rtr/np.sum(Ap*p)
    assert(1==0)
    V=V+alpha*p#pad(alpha*p)
    rnew=r-alpha*Ap
    beta=np.sum(rnew*rnew)/rtr
    p=rnew+beta*p
    r=rnew
    plt.clf();
    plt.imshow(r)
    plt.colorbar()
    plt.pause(0.001)
rho=V[1:-1,1:-1]-(V[1:-1,0:-2]+V[1:-1,2:]+V[:-2,1:-1]+V[2:,1:-1])/4.0

assert(1==0)
#bc[n//2,n//4:(3*n)//4]=1

V=bc.copy()

for i in range(2*n):
    V[1:-1,1:-1]=(V[1:-1,0:-2]+V[1:-1,2:]+V[:-2,1:-1]+V[2:,1:-1])/4.0
    V[mask]=bc[mask]
    #plt.clf()
    #plt.imshow(V)
    #plt.colorbar()
    #plt.pause(0.001)
rho=V[1:-1,1:-1]-(V[1:-1,0:-2]+V[1:-1,2:]+V[:-2,1:-1]+V[2:,1:-1])/4.0
