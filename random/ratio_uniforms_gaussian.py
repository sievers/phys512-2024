import numpy as np
from matplotlib import pyplot as plt

u=np.linspace(0,1,2001)
u=u[1:]
#have u<exp(-0.25*r^2) so
#log(u)=-0.25(v/u)^2
#v=2u*(-log(u))**0.5

v=2*u*np.sqrt(-np.log(u))
print('max v is ',v.max())

plt.ion()
plt.figure(1)
plt.clf()
plt.plot(u,v,'k')
plt.plot(u,-v,'k')
plt.show()
plt.savefig('gauss_teardrop.png')

N=1000000
u=np.random.rand(N)
#.86 seems to be max value of v
v=(np.random.rand(N)*2-1)*0.86
r=v/u
accept=u<np.exp(-0.25*r**2)
gauss=r[accept]

a,b=np.histogram(gauss,100,range=(-3,3))
bb=0.5*(b[1:]+b[:-1])
pred=np.exp(-0.5*bb**2)/np.sqrt(2*np.pi)*np.sum(accept)*(bb[2]-bb[1])
plt.figure(2)
plt.clf()
plt.bar(bb,a,0.05)
plt.plot(bb,pred,'r')
plt.show()
plt.savefig('gauss_ratio_hist.png')
