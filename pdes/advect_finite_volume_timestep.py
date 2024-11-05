
import numpy as np
from matplotlib import pyplot as plt
n=300
rho=np.zeros(n)
rho[n//3:(2*n//3)]=1
v=1.0
dx=1.0
x=np.arange(n)*dx

plt.ion()
plt.clf()
plt.axis([0,n,0,1.1])
plt.plot(x,rho)
plt.draw()
plt.savefig('advect_initial_conditions.png')

#advect_finite_volume_timestep.py
dt=1.0
big_rho=np.zeros(n+1)
big_rho[1:]=rho
del rho  #we can delete the to save space
oversamp=200 #let's do finer timestamps
dt_use=dt/oversamp
for step in range(0,450):

    #big_rho[0]=0
    big_rho[0]=big_rho[-1]
    for substep in np.arange(0,oversamp):
        drho=big_rho[1:]-big_rho[0:-1]
        big_rho[1:]=big_rho[1:]-v*dt_use/dx*drho
        big_rho[0]=big_rho[-1]
    plt.clf()
    plt.axis([0,n,0,1.1])
    plt.plot(x,big_rho[1:])
    plt.pause(0.005)
    #plt.draw()

