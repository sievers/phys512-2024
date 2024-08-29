import numpy as np
from matplotlib import pyplot as plt
plt.ion()

x0=3;
logdx=np.linspace(-15,-1,151)
dx=10**logdx

f0=np.exp(x0)
f1=np.exp(x0+dx)
my_deriv=(f1-f0)/dx

ans=np.exp(x0)
plt.clf()
plt.plot(logdx,np.log10(np.abs(my_deriv-ans)))
plt.show()

fminus=np.exp(x0-dx)
deriv2=(f1-fminus)/(2*dx)
plt.plot(logdx,np.log10(np.abs(deriv2-ans)))
plt.xlabel('Log10(dx)',fontsize=16)
plt.ylabel('Error in deriv for exp(x)|x=3',fontsize=16)
plt.legend(['First-order deriv','Second-order deriv'])
plt.title('Roundoff vs. Analytic Deriv Errors',fontsize=18)

plt.savefig('deriv_errs.png')
