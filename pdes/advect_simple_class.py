import numpy as np
from matplotlib import pyplot as plt
plt.ion()

x=np.arange(300)
rho=0*x
rho[np.abs(x-150)<50]=1
plt.clf()
plt.plot(x,rho)
plt.show()

alpha=1
for i in range(300):
    rho_new=0*rho
    rho_new[1:]=rho[1:]-alpha*rho[1:]+alpha*rho[:-1]
    plt.clf()
    plt.plot(rho_new)
    plt.pause(0.01)
    rho=rho_new
    

