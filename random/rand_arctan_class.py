import numpy as np
from matplotlib import pyplot as plt
plt.ion()
#first step - generate lorentzians


y=np.linspace(-np.pi/2,np.pi/2,1001)
y=0.5*(y[:-1]+y[1:])

height=np.exp(-0.5*np.tan(y)**2)/np.cos(y)**2
#plt.clf()
#plt.plot(y,height)
#plt.show()


n=100000
nums=np.random.rand(n)
#cdf is arctan, so inverse(cdf) is tangent

y=(nums-1/2)*np.pi
#vec=np.exp(-0.5*np.tan(y)**2)/np.cos(y)**2
vec=np.exp(-np.abs(np.tan(y)))/np.cos(y)**2
accept=vec/(1.2*height.max())
#gdevs=np.tan(vec[np.random.rand(n)<accept]) #this was the line I had wrong 
gdevs=np.tan(y[np.random.rand(n)<accept])    #this is the correct one, vec->y

bins=np.linspace(-5,5,101)
a,b=np.histogram(gdevs,bins)
bb=(bins[:-1]+bins[1:])/2

plt.clf()
plt.plot(bb,a/a.max())
plt.plot(bb,np.exp(-0.5*bb**2))
plt.plot(bb,np.exp(-np.abs(bb)))
