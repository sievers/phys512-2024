
import numpy
from matplotlib import pyplot as plt
n=300
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
plt.savefig('advect_initial_conditions.png')

#advect_finite_volume_guard.py
dt=1.0
for step in range(0,250):
    #we need one guard cell - make a 1-larger temp array
    big_rho=numpy.zeros(n+1)
    big_rho[1:]=rho
    #explicitly set the density of the guard cell
    #big_rho[0]=0
    big_rho[0]=big_rho[-1]
    #take the difference in densities
    drho=big_rho[1:]-big_rho[0:-1]
    big_rho[1:]=big_rho[1:]-v*dt/dx*drho
    rho=big_rho[1:]
    plt.clf()
    plt.axis([0,n,0,1.1])
    plt.plot(x,rho)
    plt.draw()
    plt.pause(0.001)

