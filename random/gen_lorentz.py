import numpy as np
from matplotlib import pyplot as plt

n=10000000
r=np.random.rand(n)
x=np.tan(np.pi*(r-0.5))
bins=np.linspace(-3,3,301)
db=bins[1]-bins[0]
a,b=np.histogram(x,bins)
a=a/n/db
plt.ion()
plt.clf()
bb=0.5*(bins[:-1]+bins[1:])
plt.plot(bb,a)
plt.plot(bb,1/(1+bb**2)/np.pi)
plt.show()

