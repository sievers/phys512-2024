import numpy as np

x=np.linspace(-1,1,21)
y=np.exp(x)

dx=x[1]-x[0]

myint=(y[0]/2+y[-1]/2+np.sum(y[1:-1]))*dx
myint2=(y[0]/2+y[-1]/2+np.sum(y[2:-1:2]))*(2*dx)
myint3=(4*myint-myint2)/3
truth=np.exp(x[-1])-np.exp(x[0])
print(myint,myint2,myint3,myint-truth,myint3-truth)


