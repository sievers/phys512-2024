import numpy as np
from matplotlib import pyplot as plt

def f(x,y,k=1):
    dydx=np.asarray([y[1],-k*y[0]])
    return dydx
def rk2(f,x,y,h):
    f0=f(x,y)
    k1=h*f0
    k2=h*f(x+h,y+k1)
    dy=(k1+k2)/2
    return y+dy
def rk2b(f,x,y,h):
    f0=f(x,y)
    k1=h*f0
    k2=h*f(x+h/2,y+k1/2)
    dy=k2
    return y+dy

plt.ion()

y0=np.asarray([1,0])

h=0.1/2
nstep=500*2
y=np.zeros([nstep,len(y0)])
y[0,:]=y0
x=np.arange(nstep)*h
for i in range(1,nstep):
    y[i,:]=rk2b(f,x[i-1],y[i-1,:],h)
y_true=np.cos(x)
plt.clf()
#plt.plot(x,y[:,0])
#plt.plot(x,y_true)
plt.plot(x,y[:,0]-y_true)
plt.show()
