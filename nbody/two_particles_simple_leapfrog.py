import numpy
from matplotlib import pyplot as plt
#let's start two partcles in what should be a circular orbit
x0=0;y0=0;vx0=0;vy0=0.5
x1=1;y1=0;vx1=0;vy1=-0.5;

#for simplicity, let's assume G&m are all equal to 1
dt=0.01
tmax=25
dprint=1000
dt=dt/dprint
plt.ion()
pold=0
kinold=0

step=0
plt.clf()
for t in numpy.arange(0,tmax,dt):    
    step+=1
    dx=x0-x1
    dy=y0-y1
    rsquare=dx*dx+dy*dy
    r=numpy.sqrt(rsquare)
    r3=r*r*r
    #calculate the x and y  force components
    fx0=dx/r3;
    fy0=dy/r3
    #forces on particle 1 must be opposite of particle 0
    fx1 =-fx0
    fy1 =-fy0
    #update particle positions
    x0 +=dt*vx0
    y0 +=dt*vy0
    vx0 +=-dt*fx0
    vy0 +=-dt*fy0
    
    x1+= dt*vx1
    y1+=dt*vy1
    vx1 -= dt*fx1
    vy1 -= dt*fy1
    

    if np.abs(t%0.01)<1e-6:
        #plt.clf()
        plt.plot(x0,y0,'rx')
        plt.plot(x1,y1,'b*')
        plt.ylim(-1.5,1.5)
        plt.xlim(-1,2)    
        plt.pause(0.001)

    pot=-1.0/r
    kin=0.5*(vx0*vx0+vy0*vy0+vx1*vx1+vy1*vy1)
    if (step%dprint)==0:
        print('kin and pot are '  + repr(kin) + '   ' + repr(pot) + '  ' + repr(0.5*(kin+kinold)+pot) + ' ' + repr(0.5*(pot+potold)+kin))
    potold=pot
    kinold=kin


