#advect_lagrangian.py
import numpy
from matplotlib import pyplot as plt

n=300
#set up density the usual way
rho=numpy.zeros(n)
rho[n//3:(2*n//3)]=1

v=1.0
dx=1.0
x=numpy.arange(n)*dx

plt.ion()
plt.clf()
plt.axis([0,n,0,1.1])
plt.plot(x,rho)
plt.draw()

dt=1.0
#now take time steps
for step in range(0,150):
    #new particle position is just old position plus velocity
    x=x+v*dt 
    plt.clf()
    plt.axis([0,1.5*n,0,1.1])
    plt.plot(x,rho,'*')
    #plt.draw()
    plt.pause(0.01)

    
