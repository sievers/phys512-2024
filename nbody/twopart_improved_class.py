import numpy as np
from matplotlib import pyplot as plt
plt.ion()

x0=np.asarray([1.,0])
y0=np.asarray([-1.,0])
vx0=np.asarray([0.0,0.05])
vy0=np.asarray([0.0,-0.05])


dt=0.01
plt.clf()
for t in np.arange(0,20,dt):
    dx=x0-y0
    rsqr=np.sum(dx**2)
    acc=dx/rsqr**(3/2)
    
    xf=x0+dt*vx0
    yf=y0+dt*vy0
    vxf=vx0-dt*acc
    vyf=vy0+dt*acc

    xeff=(xf+x0)/2
    yeff=(yf+y0)/2

    dxeff=xeff-yeff
    rsqr=np.sum(dxeff**2)
    acceff=dxeff/rsqr**(3/2)

    vxf=vx0-acceff*dt
    vyf=vy0+acceff*dt

    x0=x0+dt*(vx0+vxf)/2
    y0=y0+dt*(vy0+vyf)/2
    vx0=vxf
    vy0=vyf
    
    plt.plot(x0[0],x0[1],'b.')
    plt.plot(y0[0],y0[1],'r.')
    plt.pause(0.001)
