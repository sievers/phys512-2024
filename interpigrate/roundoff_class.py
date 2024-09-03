import numpy as np
from matplotlib import pyplot as plt
plt.ion()

logdx=np.linspace(-16,0,33)
dx=10**logdx

x0=100*np.pi
xx=x0+dx
dx=xx-x0


f_plus=np.exp(x0+dx)
f0=np.exp(x0)
deriv=(f_plus-f0)/dx
dtrue=np.exp(x0)

err=np.abs(deriv-dtrue)
plt.clf()
plt.plot(logdx,np.log10(err),'*')
plt.show()

f_minus=np.exp(x0-dx)
deriv=(f_plus-f_minus)/(2*dx)
err_2nd=deriv-dtrue
plt.plot(logdx,np.log10(err_2nd),'*')
