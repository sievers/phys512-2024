import numpy as np
from matplotlib import pyplot as plt


def get_forces(x,soft=0.01,do_pot=False,fail=False):
    f=np.zeros(x.shape)
    nx=x.shape[0]
    pot=0
    for i in range(nx):
        dx=x[:,0]-x[i,0]
        dy=x[:,1]-x[i,1]        
        rsqr=dx**2+dy**2
        if False:
            tmp=rsqr[rsqr>0]
            if (tmp.min()<soft**2):
                print(rsqr)
                if fail:
                    assert(1==0)
        rsqr[rsqr<soft**2]=soft**2
        r3=1/(rsqr*np.sqrt(rsqr))
        fx=dx*r3
        fy=dy*r3
        f[i,0]=np.sum(fx)
        f[i,1]=np.sum(fy)
        if do_pot:
            pot=pot+np.sum(1/rsqr)-1/soft**2 #extra bit is to remove self-energy
    if do_pot:
        return f,np.sqrt(pot/2)
    else:
        return f

#assert(1==0)


def take_step_simple(x,v,dt):
    f=get_forces(x)
    x=x+dt*v
    v=v+dt*f
    return x,v

def take_step_better(x,v,dt):
    xx=x+0.5*v*dt
    f=get_forces(xx)
    vv=v+0.5*dt*f
    x=x+dt*vv
    v=v+dt*f
    return x,v


def get_derivs(xx):
    nn=xx.shape[0]//2
    x=xx[:nn,:]
    v=xx[nn:,:]
    f=get_forces(x)
    return np.vstack([v,f])


def take_step_rk4(x,v,dt):

    xx=np.vstack([x,v])
    k1=get_derivs(xx)
    k2=get_derivs(xx+k1*dt/2)
    k3=get_derivs(xx+k2*dt/2)
    k4=get_derivs(xx+k3*dt)
    
    tot=(k1+2*k2+2*k3+k4)/6
    

    nn=x.shape[0]
    x=x+tot[:nn,:]*dt
    v=v+tot[nn:,:]*dt
    return x,v

x=np.zeros([2,2])
v=np.zeros([2,2])
x[0,0]=1
x[1,0]=-1
v[0,1]=0.25
v[1,1]=-0.25
v=0.5*v

plt.ion()
plt.clf()
soft=0.01

nfine=20
dt=soft**1.5*0.05*1100/nfine
#soft=soft/4
nstep=2000

all_pot=np.empty(nstep)
all_kin=np.empty(nstep)
plt.clf()
for i in range(nstep):
    for j in range(nfine):
        #x,v=take_step_better(x,v,dt)
        x,v=take_step_rk4(x,v,dt)
        #x,v=take_step_simple(x,v,dt)
    f,pot=get_forces(x,do_pot=True,fail=True)
    kin=np.sum(v**2)/2
    all_pot[i]=pot
    all_kin[i]=kin
    L=np.sum(x[:,0]*v[:,1]-x[:,1]*v[:,0])
    print(kin,pot,kin-pot,L)

    plt.plot(x[:,0],x[:,1],'b.')
    plt.axis([-2,2,-2,2])
    plt.pause(0.001)

    


